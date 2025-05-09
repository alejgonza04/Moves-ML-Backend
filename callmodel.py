# callmodel.py

import pandas as pd
import numpy as np
import os
import random
import time 
import joblib
import lightgbm as lgb
import requests
from functools import lru_cache                             # ADDED
from concurrent.futures import ThreadPoolExecutor, as_completed  # ADDED
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv

load_dotenv()
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# ADDED: reuse HTTP connections for keep-alive
session = requests.Session()

shopping_types = [
  "clothing_store", "shoe_store",
  "jewelry_store", "thrift_store", "record_store", "comic_book_store", "farmers_market",
]

mood_type_map = {
    "shopping": shopping_types,
    "coffee" : ["cafe", "bakery",],
    "party" : ["night_club", "bar", "pub", "brewery",],
    "food": ["restaurant",],
    "workout" : ["gym",],
    "adventure": ["amusement_park", "aquarium", "zoo", "park", "tourist_attraction", "campground", ],
    "artsy": ["art_gallery", "museum", "movie_theater", ],
    "study": ["library", "university", "book_store", ]
}

_, cv, _ = joblib.load(os.path.join(os.path.dirname(__file__), "recommender_model.joblib"))

# ADDED: in-memory cache for TF-IDF transforms
_transform_cache = {}

def fetch_google_business(name: str, city: str) -> dict:
    query = f"{name} {city}"

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        "query": f"{name} {city}",
        "key": GOOGLE_PLACES_API_KEY,
    }

    response = session.get(url, params=params)  # CHANGED to use session
    results = response.json().get("results", [])
    return results[0] if results else None

def fetch_similar_google_places(city: str, category_keywords: str, max_pages: int = 3):
  #Fetch up to max_pages * 20 results via textsearch + proper pagination.
  url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
  # initial query params (only the first time)
  params = {
      "query": f"{category_keywords} in {city}",
      "key": GOOGLE_PLACES_API_KEY,
      "fields": "name,geometry,types,rating,place_id,formatted_address,photos"
  }
  all_results = []
  token = None

  for page in range(max_pages):
      # if we have a token (2nd+ page), switch to pagetoken params
      if page > 0:
          # Google requires a short wait before next_page_token becomes valid
          time.sleep(2)
          params = {"pagetoken": token, "key": GOOGLE_PLACES_API_KEY}

      resp = session.get(url, params=params).json()
      results = resp.get("results", [])
      all_results.extend(results)

      token = resp.get("next_page_token")
      # stop early if no more pages
      if not token:
          break

  return all_results

def build_google_tag(business: dict) -> str:
    name = business.get("name", "")
    city = business.get("formatted_address", "").split(",")[-3].strip() if ","  in business.get("formatted_address", "") else ""
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

@lru_cache(maxsize=256)  # ADDED caching geocode
def get_city_from_coordinates(lattitude: float, longitude: float) -> str:
  url = "https://maps.googleapis.com/maps/api/geocode/json"
  response = session.get(url, params={  # CHANGED to use session
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
    response = session.post(url, json={})  # CHANGED to use session
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

    # ADDED: parallel fetch per category
    all_businesses = []
    with ThreadPoolExecutor(max_workers=min(10, len(mood_types))) as ex:
        futures = [ex.submit(fetch_similar_google_places, city, pt, 3) for pt in mood_types]
        for f in as_completed(futures):
            all_businesses.extend(f.result())

    # tag and vectorize the nearby businesses'''
    tagged = []
    for business in all_businesses:
        # ADDED: skip any place without photos
        if not business.get("photos"):
            continue
        rating = business.get("rating", 0)
        if rating >= rating_threshold:
            tag = build_google_tag(business)
            tagged.append((business, tag))

    if not tagged:
        return JSONResponse(status_code=200, content={"recommendations": []})

    tags = [tag for _, tag in tagged]
    # ADDED: cache TF-IDF transforms
    tag_vectors = []
    for t in tags:
        vec = _transform_cache.get(t)
        if vec is None:
            vec = cv.transform([t]).toarray()[0]
            _transform_cache[t] = vec
        tag_vectors.append(vec)
    tag_vectors = np.array(tag_vectors)

    # generate fake base vvector for similarity since we won't sort by it primarily
    dummy_vector = np.zeros_like(tag_vectors[0])
    scores = cosine_similarity([dummy_vector], tag_vectors)[0]

    combined = []
    for ((business, tag), sim_score) in zip(tagged, scores):
      business_lat = business.get("geometry", {}).get("location", {}).get("lat")
      business_lon = business.get("geometry", {}).get("location", {}).get("lng")
      if latitude is not None and longitude is not None and business_lat is not None and business_lon:
        distance_miles = haversine_distance(latitude, longitude, business_lat, business_lon) * 0.621371
      else:
        distance_miles = float('inf')  # deprioritize if no location data

      combined.append((business, sim_score, distance_miles))
    
    # within 30 mile filter
    close_options = [item for item in combined if item[2] <= 30]

    # choose one randomly from the top N cloesest, for ex top 5
    close_options.sort(key=lambda x: x[2])

    # ensures we don't have duplicates in the address
    seen_addresses = set()
    unique_nearby = []
    for business, sim_score, distance_miles in close_options:
      address = business.get("formatted_address")
      if address and address not in seen_addresses:
          seen_addresses.add(address)
          unique_nearby.append((business, sim_score, distance_miles))
    
    top_nearby = unique_nearby[:60]

    recommendations = []
    for (business, sim_score, distance_miles) in top_nearby:
      # set photo_url to none
      photo_url = None
      

      if "photos" in business and business["photos"]:
        photo_reference = business["photos"][0].get("photo_reference")
        if photo_reference:
          photo_url = (
              f"https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=400&photoreference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"
          )

        full_address = business.get("formatted_address", "")
        city = get_city_from_address(full_address)
        maps_url = f"https://www.google.com/maps/place/?q=place_id:{business.get('place_id')}"


        recommendations.append({
            "name": business.get("name"),
            "rating": business.get("rating"),
            "address": business.get("formatted_address"),
            "categories": ", ".join(business.get("types", [])),
            "distance_miles": round(distance_miles, 2),
            "image": photo_url,
            "city": city,
            "maps_url": maps_url,
            #"score": round(score, 3)
        })
    print(f"[callmodel] Generated {len(recommendations)} recommendations")
    return {"recommendations": recommendations}
