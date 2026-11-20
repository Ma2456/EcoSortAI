/**
 * DEMO DATA — a hand-written example of the /api/analyze response shape.
 * It is rendered with a "DEMO DATA" banner and never presented as a model
 * prediction. The image is a crop of the park video (metal-can bin).
 */
export const DEMO_IMAGE = '/media/sample-metal.jpg';

export const DEMO_RESPONSE = {
  request_id: 'demo',
  demo: true,
  model: { name: 'demo-sample', kind: 'demo', detector: null },
  image: { width: 400, height: 400 },
  summary: { primary_category: 'metal', primary_label: 'Metal', confidence: 0.9, uncertain: false, object_count: 1 },
  warnings: ['Demo data: values below are illustrative and were not produced by a model.'],
  timing_ms: null,
  objects: [
    {
      id: 'obj-1',
      bbox: null,
      source: 'whole-image',
      detector: null,
      prediction: {
        category: 'metal',
        category_label: 'Metal',
        raw_label: 'metal',
        confidence: 0.9,
        uncertain: false,
        top_k: [
          { category: 'metal', category_label: 'Metal', raw_label: 'metal', confidence: 0.9 },
          { category: 'plastic', category_label: 'Plastic', raw_label: 'plastic', confidence: 0.06 },
          { category: 'glass', category_label: 'Glass', raw_label: 'glass', confidence: 0.02 },
        ],
      },
      explanation: {
        summary: 'Example wording: the model would report which class it chose, how far ahead it was of the runner-up, and where in the image the Grad-CAM evidence concentrates.',
        evidence: ['Example: most activation falls on the can tops at the rim of the bin', 'Example: runner-up class is far behind'],
        limitations: ['Demo data — upload a real image with a configured model to see genuine output'],
      },
      xai: {
        gradcam: { available: false, note: 'Generated only for real analyses' },
        lime: { available: false, note: 'Generated only for real analyses' },
        shap: { available: false, note: 'Generated only for real analyses' },
      },
      recommendation: {
        bin: 'Dry waste',
        segregation: 'Empty and rinse cans. Crush aluminium cans to save space.',
        recycling: 'Aluminium and steel are melted and recast with large energy savings.',
        handling: 'Watch for sharp lid edges. Never puncture or burn aerosol cans.',
        destination: 'Dry-waste collection or scrap dealer → metal recycler.',
      },
    },
  ],
};
