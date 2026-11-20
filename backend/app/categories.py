"""The seven EcoSortAI waste categories, label mapping and disposal guidance."""
from __future__ import annotations

CATEGORIES: dict[str, dict] = {
    "plastic": {
        "label": "Plastic",
        "bin": "Dry waste",
        "segregation": "Empty, rinse and squash. Keep plastics separate from food scraps in the dry-waste bin.",
        "recycling": "PET and HDPE are widely recycled; film and multi-layer packaging need collection points that accept them.",
        "handling": "Never burn plastic. Remove food residue so the recycling batch is not contaminated.",
        "destination": "Dry-waste collection → material recovery facility → registered plastic recycler.",
        "limitation": "Clear plastic and clear glass look alike in photos.",
    },
    "paper": {
        "label": "Paper",
        "bin": "Dry waste",
        "segregation": "Keep paper and cardboard dry and free of food or oil. Flatten boxes.",
        "recycling": "Clean paper is pulped into new paper and board. Greasy or wet paper goes to compost instead.",
        "handling": "Store under cover; wet paper loses recycling value.",
        "destination": "Dry-waste collection or scrap dealer → paper mill / recycler.",
        "limitation": "Wet or food-soiled paper can resemble organic waste; laminated paper looks like paper but is not recyclable as paper.",
    },
    "metal": {
        "label": "Metal",
        "bin": "Dry waste",
        "segregation": "Empty and rinse cans; crush aluminium cans. Keep aerosol cans separate and fully empty.",
        "recycling": "Aluminium and steel are melted and recast with large energy savings over virgin metal.",
        "handling": "Watch for sharp edges. Never puncture or burn aerosol cans.",
        "destination": "Dry-waste collection or scrap dealer → metal recycler.",
        "limitation": "Metallised plastic film can look like metal; aluminium vs. steel cannot be confirmed visually.",
    },
    "organic": {
        "label": "Organic",
        "bin": "Wet waste",
        "segregation": "Collect in the wet-waste bin without plastic liners; drain excess liquid.",
        "recycling": "Home or community composting, or biogas plants.",
        "handling": "Empty daily to avoid odour and pests.",
        "destination": "Wet-waste collection → composting / biomethanation facility.",
        "limitation": "Food inside packaging is two waste types; the packaging must be separated.",
    },
    "ewaste": {
        "label": "E-Waste",
        "bin": "Hazardous / e-waste",
        "segregation": "Keep separate from all other waste. Tape battery terminals. Do not dismantle at home.",
        "recycling": "Authorised recyclers recover metals and safely handle toxic components.",
        "handling": "Damaged lithium batteries can catch fire — keep away from heat and do not crush.",
        "destination": "Brand take-back programme or authorised (CPCB-registered) e-waste collection centre.",
        "limitation": "Hidden batteries are invisible to a camera; plain plastic housings may look like plastic waste.",
    },
    "glass": {
        "label": "Glass",
        "bin": "Dry waste",
        "segregation": "Rinse; remove caps and corks. Separate by colour if your collector asks.",
        "recycling": "Container glass is crushed into cullet and remelted into new bottles and jars.",
        "handling": "Wrap broken glass in paper and label it ‘sharp’ to protect waste workers.",
        "destination": "Dry-waste collection or glass bank → cullet processor.",
        "limitation": "Window glass, mirrors and ovenware look similar but are not accepted with container glass.",
    },
    "hazardous": {
        "label": "Hazardous",
        "bin": "Domestic hazardous waste",
        "segregation": "Keep in a separate closed container, in the original packaging. Never mix with wet, dry or recyclable waste.",
        "recycling": "Mostly not recyclable at home; lamps and some containers are handled by authorised facilities, medicines by take-back points.",
        "handling": "Do not pour into drains or soil, burn, or puncture. Wrap sharps; keep away from children.",
        "destination": "Municipal domestic hazardous waste collection, chemist take-back, or an authorised hazardous waste facility.",
        "limitation": "Empty chemical containers look like ordinary plastic; contents and faded labels are not visible in a photo.",
    },
}

UNKNOWN = "unknown"

# Common label spellings in public waste datasets → EcoSortAI categories.
DEFAULT_ALIASES: dict[str, str] = {
    "plastic": "plastic", "plastics": "plastic",
    "paper": "paper", "cardboard": "paper", "carton": "paper",
    "metal": "metal", "metals": "metal", "can": "metal", "aluminium": "metal", "aluminum": "metal",
    "organic": "organic", "biological": "organic", "food": "organic", "compost": "organic", "biodegradable": "organic",
    "e-waste": "ewaste", "ewaste": "ewaste", "e_waste": "ewaste", "electronic": "ewaste", "battery": "ewaste", "electronics": "ewaste",
    "glass": "glass", "white-glass": "glass", "green-glass": "glass", "brown-glass": "glass",
    "hazardous": "hazardous", "hazard": "hazardous", "hazardous-waste": "hazardous", "chemical": "hazardous", "chemicals": "hazardous",
    "medical": "hazardous", "medicine": "hazardous", "pharmaceutical": "hazardous", "paint": "hazardous", "pesticide": "hazardous",
    "light-bulb": "hazardous", "lamp": "hazardous", "syringe": "hazardous",
    # Textile is not an EcoSortAI category; clothing labels are reported as unknown.
    "textile": UNKNOWN, "textiles": UNKNOWN, "clothes": UNKNOWN, "clothing": UNKNOWN, "shoes": UNKNOWN, "fabric": UNKNOWN,
    "trash": UNKNOWN, "other": UNKNOWN, "unknown": UNKNOWN, "background": UNKNOWN,
}


def category_label(cat: str) -> str:
    return CATEGORIES[cat]["label"] if cat in CATEGORIES else "Unknown"


def resolve_category(raw_label: str, mapping: dict[str, str] | None = None) -> str:
    key = raw_label.strip().lower()
    if mapping and raw_label in mapping:
        return mapping[raw_label]
    if mapping and key in mapping:
        return mapping[key]
    return DEFAULT_ALIASES.get(key, UNKNOWN)


def recommendation(cat: str) -> dict:
    if cat not in CATEGORIES:
        return {
            "bin": None,
            "segregation": "The model could not confidently identify this item. Check it by hand before choosing a bin.",
            "recycling": "If it is clean and dry, it is usually dry waste; if it is food-soiled, treat it as wet waste.",
            "handling": "Treat batteries, chemicals, sharp or medical items as hazardous and keep them separate.",
            "destination": "When in doubt, ask your local waste collector rather than mixing it into a recycling stream.",
        }
    c = CATEGORIES[cat]
    return {k: c[k] for k in ("bin", "segregation", "recycling", "handling", "destination")}
