import numpy as np
import os
import time 
import joblib
import requests
from functools import lru_cache                            
from concurrent.futures import ThreadPoolExecutor, as_completed 
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

load_dotenv()
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# reuse HTTP connections for keep-alive
session = requests.Session()

shopping_types = [
  "clothing_store",
  "jewelry_store", "thrift_store", "record_store", "comic_book_store", "vintage store", "local boutique", "indie store",
]

mood_type_map = {
    "shopping": shopping_types,
    "coffee" : ["cafe", "bakery", "coffee roastery","art cafe",],
    "party" : ["night_club", "bar", "karaoke", "rooftop bar", "speakeasy", "dive bar"],
    "food": ["restaurant", ],
    "workout" : ["gym", "fitness center", "crossfit", "yoga studio", "martial arts",],
    "adventure": ["amusement_park", "aquarium", "zoo", "park", "tourist_attraction", 
    "campground", "hiking", "kayaking", "boat tour", "rock climbing",],
    "artsy": ["art_gallery", "museum", "movie_theater", "indie theater", "community gallery", "street art", "ceramics studio", "art class"],
    "study": ["library", "book_store", "co-working space", "university library", "quiet cafe", "study cafe",]
}

_, cv, _ = joblib.load(os.path.join(os.path.dirname(__file__), "recommender_model.joblib"))

#in-memory cache for TF-IDF transforms
_transform_cache = {}

def fetch_google_business(name: str, city: str) -> dict:
    query = f"{name} {city}"

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        "query": f"{name} {city}",
        "key": GOOGLE_PLACES_API_KEY,
    }

    response = session.get(url, params=params)  #  use session
    results = response.json().get("results", [])
    return results[0] if results else None

def fetch_similar_google_places(city: str, keyword: str, max_pages: int = 3):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{keyword} in {city}",
        "key": GOOGLE_PLACES_API_KEY,
        "fields": "name,geometry,types,rating,place_id,formatted_address,photos"
    }
    all_results = []
    token = None

    for page in range(max_pages):
        if page == 0:
            resp = session.get(url, params=params).json()
        else:
            # poll until the next_page_token is active
            for _ in range(5):
                poll = session.get(url, params={
                    "pagetoken": token,
                    "key": GOOGLE_PLACES_API_KEY
                }).json()
                if poll.get("results"):
                    resp = poll
                    break
                time.sleep(0.5)
            else:
                break  # bail if token never activates

        all_results.extend(resp.get("results", []))
        token = resp.get("next_page_token")
        if not token:
            break

    return all_results

def build_google_tag(business: dict) -> str:
    name = business.get("name", "")
    address = business.get("formatted_address", "")
    parts = address.split(",")
    city = parts[-3].strip() if len(parts) >= 3 else ""
    #city = business.get("formatted_address", "").split(",")[-3].strip() if ","  in business.get("formatted_address", "") else ""
    categories = " ".join(business.get("types", []))
    return f"{name} {categories} {city}"

# calculate longitude and lattitude
def haversine_distance(lat1, lon1, lat2, lon2):
  R = 6371  # Radius of the Earth in kilometer
  lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
  dlat = lat2 - lat1
  dlon = lon2 - lon1
  a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
  return R * 2 * asin(sqrt(a))

@lru_cache(maxsize=256)  # caching geocode
def get_city_from_coordinates(lattitude: float, longitude: float) -> str:
  url = "https://maps.googleapis.com/maps/api/geocode/json"
  response = session.get(url, params={  # to use session
      "latlng": f"{lattitude},{longitude}",
      "key": GOOGLE_PLACES_API_KEY,
  })
  results = response.json().get("results", [])
  if not results:
    return ""

  for component in results[0].get("address_components", []):
    if "locality" in component.get("types", []):
        return component.get("long_name")
  return ""
 
def fetch_location_and_mood_from_flask():
  url = "http://localhost:5555/locationmood"
  try:
    response = session.post(url, json={})  # to use session
    if response.status_code == 200:
      data = response.json()
      print(data)
      return data
  except Exception as e:
    print("Exception fetching location/mood:", e)
  return None
  
def get_city_from_address(address: str) -> str:
  parts = address.split(",")
  if len(parts) >= 3:
      return parts[-3].strip()
  return ""

def get_recommendations(user_mood: str = None, latitude: float = None, longitude: float = None, rating_threshold: float = 3.0):
   # get base business from user
    if latitude is None or longitude is None:
      location_data = fetch_location_and_mood_from_flask()
      if location_data:
        latitude = location_data.get("latitude")
        longitude = location_data.get("longitude")
        user_mood = user_mood or location_data.get("mood")
    
    print(f"User location: ({latitude}, {longitude}), mood: {user_mood}")

    city = get_city_from_coordinates(latitude, longitude)
    mood_types = mood_type_map.get(user_mood.lower())

    # parallel fetch per category
    all_businesses = []
    with ThreadPoolExecutor(max_workers=min(10, len(mood_types))) as ex:
        futures = [ex.submit(fetch_similar_google_places, city, pt, 3) for pt in mood_types]
        for f in as_completed(futures):
            all_businesses.extend(f.result())

    # tag and vectorize the nearby businesses'''
    tagged = []
    for business in all_businesses:
        # skip any place without photos
        if not business.get("photos"):
            continue
        rating = business.get("rating", 0)
        if rating >= rating_threshold:
            tag = build_google_tag(business)
            tagged.append((business, tag))

    if not tagged:
        return JSONResponse(status_code=200, content={"recommendations": []})

    tags = [tag for _, tag in tagged]
    # identify only the new tags to transform
    new_tags = [t for t in tags if t not in _transform_cache]
    if new_tags:
        # batch-transform
        new_vectors = cv.transform(new_tags).toarray()
        for t, v in zip(new_tags, new_vectors):
            _transform_cache[t] = v

    # assemble tag_vectors from cache
    tag_vectors = np.stack([_transform_cache[t] for t in tags])

    # generate similarity against a zero‐vector
    scores = cosine_similarity(np.zeros((1, tag_vectors.shape[1])), tag_vectors)[0]

    combined = []
    with ThreadPoolExecutor() as dist_ex:
        dist_futs = []
        for (business, _), sim_score in zip(tagged, scores):
            loc = business.get("geometry", {}).get("location", {})
            if "lat" in loc and "lng" in loc:
                dist_futs.append(
                    dist_ex.submit(
                        lambda b, s: (b, s,
                          haversine_distance(latitude, longitude,
                            b["geometry"]["location"]["lat"],
                            b["geometry"]["location"]["lng"]) * 0.621371),
                        business,
                        sim_score
                    )
                )
        for fut in as_completed(dist_futs):
            combined.append(fut.result())

    # within 30 mile filter
    close_options = [item for item in combined if item[2] <= 50]
    close_options.sort(key=lambda x: x[2])

    # ensure unique addresses
    seen = set()
    unique_nearby = []
    for business, sim_score, distance_miles in close_options:
        addr = business.get("formatted_address")
        if addr and addr not in seen:
            seen.add(addr)
            unique_nearby.append((business, sim_score, distance_miles))

    top_nearby = unique_nearby[:60]

    recommendations = []
    for business, sim_score, distance_miles in top_nearby:
        photo_url = None
        photos = business.get("photos", [])
        if photos:
            ref = photos[0].get("photo_reference")
            if ref:
                photo_url = (
                    f"https://maps.googleapis.com/maps/api/place/photo"
                    f"?maxwidth=400&photoreference={ref}&key={GOOGLE_PLACES_API_KEY}"
                )

        full_address = business.get("formatted_address", "")
        city = get_city_from_address(full_address)
        maps_url = f"https://www.google.com/maps/place/?q=place_id:{business.get('place_id')}"

        recommendations.append({
            "name": business.get("name"),
            "rating": business.get("rating"),
            "address": full_address,
            "categories": ", ".join(business.get("types", [])),
            "distance_miles": round(distance_miles, 2),
            "image": photo_url,
            "city": city,
            "maps_url": maps_url,
        })

    print(f"[callmodel] Generated {len(recommendations)} recommendations")
    return {"recommendations": recommendations}