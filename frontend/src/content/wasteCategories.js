/**
 * Educational content for the seven bin stations.
 *
 * This is predefined teaching material, NOT model output. Anything that looks
 * like a model result (class, confidence, Grad-CAM/LIME/SHAP) is an
 * illustration of what the explanation means, and the UI labels it as such.
 * Real predictions only appear in the Waste Lab when the backend has a model.
 */

export const CATEGORIES = {
  plastic: {
    name: 'Plastic Waste',
    short: 'Plastic',
    stream: 'Dry waste · mostly recyclable',
    tagline: 'Light, glossy and everywhere — but only some of it gets a second life.',
    accent: '#7fb2d9',
    identify: {
      examples: ['PET water bottles', 'HDPE milk & shampoo bottles', 'Food containers & lids', 'Carry bags & wrappers', 'Bottle caps'],
      characteristics: ['Specular highlights and transparency', 'Moulded, smooth curves', 'Crinkled thin film', 'Resin code triangle (1–7)'],
      subcategories: ['PET (1)', 'HDPE (2)', 'LDPE film (4)', 'PP (5)', 'Multi-layer packaging'],
    },
    ai: {
      predictedClass: 'plastic',
      exampleConfidence: 0.91,
      cues: ['Translucent body with sharp specular reflections', 'Bottle neck & cap silhouette', 'Thin, irregular folds on film plastic'],
      process:
        'A convolutional network converts the photo into layers of features — edges, then textures, then object parts. The final layer scores each of the waste categories; a softmax turns the scores into probabilities.',
    },
    xai: {
      gradcam: 'Heat concentrates on the bottle neck, cap and label band — the shape cues that separate PET from glass.',
      lime: 'Superpixels covering the transparent body and cap push the prediction toward plastic; grass and pavement around it are neutral.',
      shap: [
        { feature: 'Transparent body', weight: 0.34 },
        { feature: 'Cap & neck shape', weight: 0.27 },
        { feature: 'Crinkled surface', weight: 0.12 },
        { feature: 'Green background', weight: -0.04 },
      ],
      plain: 'The model is mostly looking at the shape of the neck and how light passes through the body. That is sensible evidence — and also why clear glass bottles are its most common confusion.',
      limitations: ['Clear glass and clear plastic look alike in photos', 'Crushed or dirty bottles lose their shape cues', 'A photo cannot read the resin code reliably'],
    },
    action: {
      segregation: 'Empty, rinse and squash. Keep caps on. Put clean plastics in the dry-waste bin, separate from food scraps.',
      recycling: 'PET and HDPE are widely recycled into fibre, strapping and new containers. Thin film and multi-layer pouches go to collection points that accept them.',
      handling: 'Do not burn plastic — it releases toxic fumes. Remove food residue so the batch is not contaminated.',
      destination: 'Dry-waste collection → material recovery facility (MRF) → registered plastic recycler.',
    },
  },

  paper: {
    name: 'Paper Waste',
    short: 'Paper',
    stream: 'Dry waste · recyclable if clean',
    tagline: 'Fibres that can be pulped five to seven times — if they stay dry and clean.',
    accent: '#c9b98a',
    identify: {
      examples: ['Newspapers & magazines', 'Corrugated cardboard boxes', 'Office paper & notebooks', 'Paper bags', 'Egg cartons'],
      characteristics: ['Matte, fibrous texture', 'Printed text and halftone patterns', 'Folded flat edges', 'Corrugated flute lines on cardboard'],
      subcategories: ['Newsprint', 'Cardboard / OCC', 'Mixed office paper', 'Tetra Pak (paper-based composite)'],
    },
    ai: {
      predictedClass: 'paper',
      exampleConfidence: 0.88,
      cues: ['Dense printed text columns', 'Flat sheets with straight folds', 'Brown corrugated edges'],
      process:
        'Texture-sensitive filters in the early layers respond strongly to print and fibre patterns; deeper layers combine them with the flat, folded geometry typical of paper and board.',
    },
    xai: {
      gradcam: 'Activation sits on the printed newspaper surface and the exposed corrugated edge of the cardboard.',
      lime: 'Text-covered superpixels support “paper”; a superpixel containing a glossy laminated flap pulls toward plastic.',
      shap: [
        { feature: 'Printed text texture', weight: 0.31 },
        { feature: 'Corrugated edge', weight: 0.22 },
        { feature: 'Flat folded sheets', weight: 0.15 },
        { feature: 'Glossy coating', weight: -0.09 },
      ],
      plain: 'Print and flat folds are the main evidence. Glossy or laminated paper weakens the decision because the surface starts to look like plastic.',
      limitations: ['Wet or food-soiled paper looks like organic waste', 'Laminated & wax-coated paper is not recyclable but looks like paper', 'Tetra Pak cartons are often misclassified'],
    },
    action: {
      segregation: 'Keep paper dry and free of food and oil. Flatten boxes. Remove plastic windows and tape where you can.',
      recycling: 'Clean paper and cardboard are pulped into new board and paper. Greasy pizza boxes and tissues go with wet/compost waste instead.',
      handling: 'Store under cover — wet paper loses fibre quality and value.',
      destination: 'Dry-waste collection or local kabadiwala → paper mill / recycler.',
    },
  },

  metal: {
    name: 'Metal Waste',
    short: 'Metal',
    stream: 'Dry waste · highly recyclable',
    tagline: 'Aluminium cans can return to shelves as new cans, again and again.',
    accent: '#b7bfc4',
    identify: {
      examples: ['Aluminium beverage cans', 'Steel food tins', 'Aerosol cans (empty)', 'Foil trays', 'Metal lids & bottle caps'],
      characteristics: ['Metallic sheen and hard reflections', 'Cylindrical form with ridged rims', 'Pull-tabs and seams', 'Printed branding wrapped around a curve'],
      subcategories: ['Aluminium (non-ferrous)', 'Steel / tin-plate (ferrous)', 'Mixed scrap metal'],
    },
    ai: {
      predictedClass: 'metal',
      exampleConfidence: 0.93,
      cues: ['High-contrast specular streaks along a cylinder', 'Ring-pull and rolled rim', 'Dented, crumpled metal folds'],
      process:
        'The model has learned that metal reflects light in long, sharp streaks that bend around curved surfaces — very different from the softer highlights on plastic.',
    },
    xai: {
      gradcam: 'Heat clusters on can tops and rims — the ring-pull and rolled edge are the strongest evidence.',
      lime: 'Superpixels with bright metallic streaks support “metal”; the grey bin body itself contributes little.',
      shap: [
        { feature: 'Ring-pull / rim', weight: 0.36 },
        { feature: 'Metallic reflections', weight: 0.29 },
        { feature: 'Cylindrical shape', weight: 0.14 },
        { feature: 'Grey bin body', weight: 0.02 },
      ],
      plain: 'The model relies on the can tops and the way light streaks across metal. It is not using the bin colour — which is exactly what we want.',
      limitations: ['Metallised plastic film (chip packets) can fool it', 'Cannot tell aluminium from steel visually with certainty', 'Pressurised or chemical containers need special handling regardless of class'],
    },
    action: {
      segregation: 'Empty and rinse cans. Crush aluminium cans to save space. Keep aerosols separate and fully empty.',
      recycling: 'Aluminium and steel are melted and recast with large energy savings compared with virgin metal.',
      handling: 'Watch for sharp lid edges. Never puncture or burn aerosol cans.',
      destination: 'Dry-waste collection or scrap dealer → metal recycler / smelter.',
    },
  },

  organic: {
    name: 'Organic Waste',
    short: 'Organic',
    stream: 'Wet waste · compostable',
    tagline: 'Kitchen scraps are not trash — they are tomorrow’s soil.',
    accent: '#c08a55',
    identify: {
      examples: ['Fruit & vegetable peels', 'Leftover cooked food', 'Tea leaves & coffee grounds', 'Eggshells', 'Garden leaves & flowers'],
      characteristics: ['Irregular, natural shapes', 'Warm organic colours (orange, brown, green)', 'Moist, soft textures', 'Mixed, heaped arrangements'],
      subcategories: ['Kitchen / food waste', 'Garden / green waste', 'Temple flower waste'],
    },
    ai: {
      predictedClass: 'organic',
      exampleConfidence: 0.89,
      cues: ['Irregular curved peel shapes', 'Natural colour variation', 'No straight manufactured edges'],
      process:
        'Organic items have no consistent shape, so the model leans on colour distributions and soft, irregular textures rather than outlines.',
    },
    xai: {
      gradcam: 'Activation spreads across the heaped scraps on top of the bin instead of focusing on one object.',
      lime: 'Superpixels with orange and yellow peel push toward organic; a superpixel with a plastic bag in it pushes away.',
      shap: [
        { feature: 'Peel colour & texture', weight: 0.33 },
        { feature: 'Irregular shapes', weight: 0.21 },
        { feature: 'Moist surface sheen', weight: 0.11 },
        { feature: 'Plastic bag fragment', weight: -0.12 },
      ],
      plain: 'Colour and irregular texture drive this decision. A diffuse heatmap is normal for organic waste because there is no single defining object.',
      limitations: ['Food packed in plastic is ambiguous — it is two waste types', 'Soiled paper often looks organic', 'Spoiled vs. fresh food is not distinguishable'],
    },
    action: {
      segregation: 'Collect in the wet-waste bin, without plastic liners. Drain excess liquid.',
      recycling: 'Home composting, community composting or biogas plants turn this into manure and energy.',
      handling: 'Empty daily to avoid odour and pests. Keep bones, oil and meat out of small home compost bins.',
      destination: 'Wet-waste collection → composting unit / biomethanation plant.',
    },
  },

  ewaste: {
    name: 'E-Waste',
    short: 'E-Waste',
    stream: 'Hazardous · authorised recycler only',
    tagline: 'Small devices, valuable metals, toxic risks — never with household trash.',
    accent: '#9fc1a6',
    identify: {
      examples: ['Mobile phones & chargers', 'Cables & earphones', 'Circuit boards', 'Batteries (removed)', 'Remote controls & small gadgets'],
      characteristics: ['Dark plastic housings', 'Exposed wires and connectors', 'Green PCB with solder points', 'Screens, buttons and ports'],
      subcategories: ['IT & telecom equipment', 'Small consumer electronics', 'Batteries', 'Cables & accessories'],
    },
    ai: {
      predictedClass: 'e-waste',
      exampleConfidence: 0.84,
      cues: ['Cables with connector plugs', 'Rectangular devices with screens or buttons', 'PCB green with fine component detail'],
      process:
        'E-waste is visually diverse, so the model combines several part-level cues — cables, ports, screens — rather than one overall shape.',
    },
    xai: {
      gradcam: 'Hot spots fall on cable connectors and the edge of a device screen inside the open bin.',
      lime: 'Superpixels with wires and ports support “e-waste”; the dark bin lid alone does not.',
      shap: [
        { feature: 'Cables & connectors', weight: 0.3 },
        { feature: 'Screen / button panel', weight: 0.24 },
        { feature: 'Circuit board pattern', weight: 0.17 },
        { feature: 'Dark plastic housing', weight: -0.05 },
      ],
      plain: 'Connectors and screens are the decisive evidence. Lower confidence than other classes is expected — e-waste varies a lot and overlaps with hard plastic.',
      limitations: ['Plain plastic casings look like plastic waste', 'Batteries hidden inside devices are invisible to a camera', 'Swollen or damaged batteries need human inspection'],
    },
    action: {
      segregation: 'Keep e-waste separate from all other bins. Tape battery terminals. Do not dismantle devices at home.',
      recycling: 'Authorised recyclers recover copper, gold, silver and rare metals and handle toxic components safely.',
      handling: 'Damaged lithium batteries can catch fire — store them away from heat and do not crush them.',
      destination: 'Brand take-back programme or an authorised (CPCB-registered) e-waste collection centre.',
    },
  },

  glass: {
    name: 'Glass Waste',
    short: 'Glass',
    stream: 'Dry waste · endlessly recyclable',
    tagline: 'Glass can be melted and reformed indefinitely without losing quality.',
    accent: '#9ccfb3',
    identify: {
      examples: ['Beverage bottles', 'Food jars', 'Cosmetic & medicine bottles (empty)', 'Broken glassware'],
      characteristics: ['Thick, refracting walls', 'Green, brown or clear tints', 'Heavy rounded base', 'Sharp, clean edges on fragments'],
      subcategories: ['Clear (flint)', 'Green', 'Amber / brown', 'Non-recyclable glass (mirrors, window panes, Pyrex)'],
    },
    ai: {
      predictedClass: 'glass',
      exampleConfidence: 0.86,
      cues: ['Refraction that distorts the background behind it', 'Coloured tint through thick walls', 'Thick lip and heavy base'],
      process:
        'The model separates glass from plastic largely by refraction — glass bends the scene behind it — plus the tinted colour and thicker rim.',
    },
    xai: {
      gradcam: 'Heat sits on the bottle necks and the refracted, distorted background visible through the glass.',
      lime: 'Superpixels with green-tinted refraction support “glass”; the transparent bin walls themselves are weakly negative.',
      shap: [
        { feature: 'Refraction distortion', weight: 0.32 },
        { feature: 'Green / amber tint', weight: 0.25 },
        { feature: 'Thick lip & base', weight: 0.14 },
        { feature: 'Clear bin wall', weight: -0.07 },
      ],
      plain: 'Refraction and colour tint are the key evidence. Because this bin is itself transparent, the model has to ignore the container and focus on the bottles inside.',
      limitations: ['Clear PET bottles are the main confusion', 'Window glass and mirrors look like recyclable glass but are not accepted', 'Broken glass in a pile is hard to segment'],
    },
    action: {
      segregation: 'Rinse and keep separate by colour if your collector asks. Remove metal caps and corks.',
      recycling: 'Container glass is crushed into cullet and remelted into new bottles and jars.',
      handling: 'Wrap broken glass in newspaper and label it “sharp” to protect waste workers.',
      destination: 'Dry-waste collection or glass bank → cullet processor → glass manufacturer.',
    },
  },

  hazardous: {
    name: 'Hazardous Waste',
    short: 'Hazardous',
    stream: 'Domestic hazardous · keep separate',
    tagline: 'A small share of household waste that can poison soil, water and the people who handle it.',
    accent: '#e0a24a',
    identify: {
      examples: ['Paint & solvent cans', 'Pesticide & cleaner bottles', 'Expired medicines', 'CFL & tube lights', 'Used syringes & sanitary waste'],
      characteristics: ['Hazard pictograms and warning labels', 'Chemical-style bottles with child-safe caps', 'Blister packs and pill strips', 'Thin glass tubes of fluorescent lamps'],
      subcategories: ['Household chemicals', 'Pharmaceutical waste', 'Mercury-containing lamps', 'Biomedical / sanitary waste', 'Pressurised containers'],
    },
    ai: {
      predictedClass: 'hazardous',
      exampleConfidence: 0.78,
      cues: ['Hazard pictograms (flame, skull, corrosion)', 'Warning-label text blocks', 'Pill blister packs and syringe shapes'],
      process:
        'Hazardous items share few visual traits, so the model leans on small details — hazard symbols, label layouts, pill strips. Confidence is usually lower than for other categories, and uncertain results matter most here.',
    },
    xai: {
      gradcam: 'Heat should land on the hazard pictogram and warning label rather than the container colour.',
      lime: 'Superpixels containing the warning symbol support “hazardous”; the plain plastic body of the bottle pulls toward plastic.',
      shap: [
        { feature: 'Hazard pictogram', weight: 0.33 },
        { feature: 'Warning-label text', weight: 0.21 },
        { feature: 'Blister pack / syringe', weight: 0.15 },
        { feature: 'Plain plastic body', weight: -0.11 },
      ],
      plain: 'The decisive evidence is the label, not the container. If the explanation highlights only the bottle shape, the model may be guessing “plastic” for the wrong reason.',
      limitations: ['An empty chemical bottle looks like ordinary plastic', 'Labels may be missing, worn or in another language', 'Contents (e.g. leftover pesticide) are invisible in a photo'],
    },
    action: {
      segregation: 'Keep in a separate, closed container — never mix with wet, dry or recyclable waste. Keep products in their original containers.',
      recycling: 'Mostly not recyclable at home. Lamps and some containers are processed by authorised facilities; medicines go to take-back points.',
      handling: 'Do not pour chemicals into drains or soil, burn them, or puncture pressurised cans. Wrap syringes and sharps; keep out of children’s reach.',
      destination: 'Municipal domestic hazardous waste collection, chemist take-back for medicines, or an authorised hazardous waste facility.',
    },
  },
};

export const TAB_LABELS = [
  { id: 'identify', label: 'Identify' },
  { id: 'ai', label: 'How AI sees it' },
  { id: 'xai', label: 'Explainable AI' },
  { id: 'action', label: 'What to do' },
];
