Image generation rewards specificity. Vague prompt → vague image.

Structure every prompt around five questions:

1. **Subject.** What is in the frame? Use concrete nouns. "A red barn"
   beats "a building"; "a black Border Collie mid-stride" beats "a
   dog".
2. **Composition.** Wide shot vs close-up; centred vs rule-of-thirds;
   eye level vs low angle. "Low-angle wide shot" is a complete answer.
3. **Lighting.** Golden-hour, overcast, harsh midday, candlelit,
   neon. The same scene under different light is a different image.
4. **Medium / style.** Photograph (35mm? polaroid?), oil painting,
   pen-and-ink, 3D render. Name the medium; the model latches on.
5. **Mood.** Serene, ominous, joyful. One adjective is usually enough.

Negative prompt: list what you want excluded. Common items: "extra
fingers, watermark, text, blurry, low resolution, cropped".

Seeds: pass `--seed N` for reproducibility. The same prompt + same seed
yields the same image. Use this to iterate: change ONE thing per pass.

Steps and guidance:
- Steps: 20-30 is plenty for most SDXL-class models. Higher = slower,
  diminishing returns.
- Guidance (CFG scale): 5-9 is the usable range. Lower = more
  creative interpretation; higher = stricter adherence to the prompt
  (and worse artefacts past ~12).

Backends and install cost:
- The default `placeholder` backend writes a 1x1 PPM. Useful only for
  pipeline testing.
- A real backend (diffusers + SDXL, MLX-Image Gen, hosted APIs) is
  user-supplied. Install size for a local SDXL setup is ~6 GB. Hosted
  APIs avoid that but cost per generation.

Don't promise an image until you see one. Backends fail, models refuse,
seeds disappoint. State what you tried and what came out.
