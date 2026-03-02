import re

def route_message(user_message: str) -> str:
    txt = (user_message or "").lower().strip()

    if txt.startswith("api:"):
        return "api"

    if txt.startswith("docs:") or txt.startswith("search docs:"):
        return "semantic"

    if re.search(r"^\s*(convert|calculate|calc)\b", txt) or "days between" in txt:
        return "tools"

    # Default to semantic for nutrition/diet questions so it uses your Chroma KB first
    nutrition_words = [
        "diet", "meal", "plan", "calorie", "calories", "protein", "carb", "carbs", "fat", "fiber",
        "cholesterol", "diabetes", "blood pressure", "hypertension", "dash", "mediterranean",
        "low sodium", "heart", "weight loss", "weight gain", "snack",
        "grocery", "shopping list", "vegan", "vegetarian", "gluten", "lactose",
        "pcos", "thyroid", "kidney", "gout", "ibs", "gerd"
    ]
    if any(w in txt for w in nutrition_words):
        return "semantic"

    return "chat"