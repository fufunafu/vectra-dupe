# Facial regions and per-region filler response rules

Source review of the 14 unique books in `~/Downloads/books` (2026-09-08).
Purpose: divide the face into sections, each with its own rule set for how it responds to
filler, and mark every rule with where the number came from and how strong the evidence is.

Companion machine-readable table: `docs/filler-regions.json`.

Short names used in citations:

| Short | Book |
|---|---|
| ClinAnat | Kim et al., *Clinical Anatomy of the Face for Filler and Botulinum Toxin Injection*, 2nd ed. 2024 (Korean cadaver + ultrasound) |
| ArtSci | Hong et al., *The Art and Science of Filler Injection*, 2nd ed. 2025 (Korean) |
| MinInv | Lee (ed.), *Innovative Injectable Techniques in Minimally Invasive Aesthetics*, 2025 |
| AnatFFN | Bedrossian et al., *Anatomy of the Forehead, Face, and Neck*, 2024 (dissection manual) |
| Periorb | Eshraghi & Pourazizi, *Periorbital Rejuvenation*, 2026 |
| Oculo | El Toukhy (ed.), *Oculoplastic Surgery*, 2nd ed. 2024 |
| PRSFund | *Plastic and Reconstructive Surgery Fundamentals*, 2024 |
| PSCurr | Luce, *A Curriculum for Plastic Surgery*, 2025 |
| DermUS | Wortsman (ed.), *Textbook of Dermatologic Ultrasound*, 2022 |
| USProto | Malherbe & Roberts, *Ultrasound Protocol for Facial Aesthetics*, 2024 |
| USArticle | Gonzalez et al., Aesth Plast Surg 2026, ultrasound of high-risk areas |

Skipped as irrelevant: Vesalius biography, pathology textbook, ENT procedures, gift certificate.
`US protocol for facial aesthstics.pdf` is a byte-identical duplicate of `Protocol US aeshtsics.pdf`.

Evidence grades used below:

- **A** = measured numbers (cadaver, ultrasound, CT, MRI, or a stated study) with n or SD.
- **B** = expert clinical ranges stated as numbers (volumes, G' bands, timelines) without a study behind them.
- **C** = consistent qualitative consensus across two or more books.
- **none** = no book says anything.

---

## 1. The headline finding

The books are strong on *anatomy* and *technique* and almost silent on *measured surface response*.
Across roughly 5,000 pages there is:

- **One** per-region skin and superficial-fat thickness table (ClinAnat Table 1.1, p.6, Korean cadavers).
- **One** HA volume-versus-time curve (ArtSci p.54–55), not region specific.
- **Zero** 3D-imaging (Vectra, MRI, ultrasound) outcome numbers tying injected mL to surface mm in any region.
- **Zero** fat-compartment volume-loss numbers (Gierloff 2012 is cited in three books but never reproduced).

So the region rule set can be built from the books for: boundaries, layer stack, injection plane,
typical volumes, stiffness class, spread pattern, timing, mobility, persistence rank, and danger
zones. The one thing the scanner actually needs to predict, **surface displacement per mL per region**,
is not in the books and has to be fitted from our own before/after scans. Section 6 lists exactly
which parameters are priors from the literature and which must be measured.

---

## 2. Why regions behave differently: the three governing variables

Every book converges on the same mechanism, which gives the axes for the taxonomy.

**2.1 Plane determines spread pattern (C, strongest cross-book agreement).**
ClinAnat p.51–53 gives the physical model: the superficial adipose tissue (SAT) has honeycomb
vertical septa anchored to dermis, so filler there "can be localized and cannot spread easily" and
"enhance[s] the skin lifting effect" (localized projection). The deep adipose / sub-SMAS /
supraperiosteal glide plane has loose oblique septa, so filler "can spread easily" and gives "wide
smooth surface lifting without lumps". ArtSci p.176 states the same for the nasolabial fold: large
subcutaneous volumes cause "lateral expansion not projection". DermUS p.526 confirms the geometry on
ultrasound: high-G' HA on the periosteum of zygoma or chin forms oval pockets; deep-to-masseter it
meanders between muscle fibres; in deep fat pads it "stays unaltered for many months".

Scanner consequence: for the same mL, a SAT injection gives a small footprint with high peak
displacement; a deep injection gives a large footprint with low peak displacement that may sit near
the per-vertex noise floor even when the integrated volume is the same.

**2.2 Tissue over bone determines how much of the volume reaches the surface (C).**
Bone-backed thin regions (forehead 4.7 mm to bone, chin, jawline) push nearly everything outward.
Regions with deep compliant spaces absorb volume: deep-to-temporalis temple "may not show change
despite a large amount" (ArtSci p.136), deep temporal plane "needs larger volume" (ClinAnat p.120),
buccal fat "requires a large volume" (ClinAnat p.139), premental space supraperiosteal bolus "would
require large amounts" (ArtSci p.87).

**2.3 Mobility determines persistence and migration (C).**
Deep fat pads with minimal muscle movement retain filler longest (DermUS p.526). Hyperactive
mentalis migrates chin filler superiorly and shortens duration (ClinAnat p.165–167, MinInv p.179).
Orbicularis oculi pushes superficial infraorbital filler and causes late oedema (MinInv p.57).
Perioral is "very dynamic" and needs repeated sessions (ClinAnat p.169). Frontalis moves low-viscosity
forehead filler (ArtSci p.131).

---

## 3. Cross-region parameters (apply to every region unless overridden)

### 3.1 Layer thickness by region (A) — ClinAnat Table 1.1, p.6, mean ± SD, mm, Korean cadavers

| Region | Skin | Superficial fat | Skin + fat |
|---|---|---|---|
| Radix and dorsum | 1.51 ± 0.55 | 1.61 ± 1.07 | 3.1 |
| Supraorbital | 1.67 ± 0.83 | 1.82 ± 1.22 | 3.5 |
| Forehead | 1.70 ± 0.71 | 1.99 ± 1.21 | 3.7 |
| Temple | 1.65 ± 0.91 | 2.58 ± 1.68 | 4.2 |
| Cheek | 1.85 ± 1.03 | 4.54 ± 2.71 | 6.4 |
| Infraorbital | 1.97 ± 0.84 | 4.93 ± 2.98 | 6.9 |
| Perioral | 1.82 ± 0.83 | 5.14 ± 3.31 | 7.0 |

Point values (A): upper eyelid skin 0.38–0.80 mm (ClinAnat p.5), ≈0.4 mm (AnatFFN p.36); lower lid
skin 0.37 to >2 mm (MinInv p.41); ala, tip, chin skin ≈2.0 mm, >2 mm at pogonion, zygion, cheek
(ClinAnat p.5–6); facial dermis 1.0–1.5 mm (DermUS p.417). No subcutaneous fat in eyelids or lips
(DermUS p.90, AnatFFN p.47). Forehead total soft tissue 4.7 ± 0.3 mm, frontalis 2.0–3.3 mm deep
(ClinAnat p.10); frontalis 3–5 mm deep (ArtSci p.129); 2–7 mm, usually 3.5–5 (Periorb p.189).
Masseter region skin-to-bone 15–25 mm (ClinAnat p.88). Mentalis 6.7–10.6 mm deep (ClinAnat p.83).
Orbicularis oculi ≈1.56 mm thick at 0.88–2.8 mm depth (MinInv p.177). Temporoparietal fascia
2–3 mm (USProto p.31). Temple skin-to-bone in one patient 28 mm (USArticle Fig 10A).

### 3.2 HA volume-over-time curve (B, single source) — ArtSci p.54–55

| Time | Volume vs injected | Max height | Width |
|---|---|---|---|
| Week 0 | 1.0 | 1.0 | 1.0 |
| Week 4 | ≈1.8 (hydration peak) | ≈0.67 | ≈1.8 |
| Week 16 | ≈0.75 | held | ≈1.8 |
| Weeks 16–64 | ≈ initial | held | declines to ≈1.0 by week 64 |

Supporting statements: "20% saturation" water uptake (ArtSci p.57); water saturation complete at
≈2 months (MinInv p.141); free-HA fraction of biphasic products absorbed in 2–3 days (ArtSci p.57,
122); transient oedema resolves in 2–3 days (ArtSci p.200); immediate contours exaggerated,
normalise over 1–2 weeks, final at 2–4 weeks (MinInv p.77); lips lose 10–20% at 1 week (ArtSci p.181);
type III collagen ≈13.8% of filler volume at 4 weeks, 21.5% at 32 weeks (ArtSci p.54).

Scanner consequence: a scan at 1–4 weeks measures hydration and oedema, not filler. Week 4 can read
up to 1.8× the injected volume; week 16 reads ≈0.75×. Report measured volume with the scan interval,
and treat week 8–16 as the earliest "settled" comparison. The width×height numbers say the footprint
grows and the peak falls during the first month, which matters for the region detector's threshold.

### 3.3 Persistence rank by plane and product (B/C)

- Dermal placement has the shortest half-life (high hyaluronidase); deep / supraperiosteal longest
  (ArtSci p.212). Deep fat pads with minimal movement retain filler longest (DermUS p.526).
- Cross-linked HA detectable on ultrasound up to 7 years; non-cross-linked ≈6 months (DermUS p.417).
  Midface HA "persist[s] for years" on MRI (MinInv p.148). Infraorbital cross-linked HA >12 months
  (MinInv p.141). Pretarsal roll "over 3 years" (ClinAnat p.114). Clinical effect fades before the
  mass clears (DermUS p.526).
- Board-review table (PRSFund p.617): collagen 3–4 mo; HA 6–12 mo; CaHA, PLLA, PCL, fat 1–2 yr;
  PMMA, polyalkylimide permanent. Treat as conservative labelling, contradicted by the imaging data above.

### 3.4 Stiffness bands per region (B) — G' at 0.1 Hz unless noted

| Region | G' target | Source |
|---|---|---|
| Jawline | > 400 Pa | MinInv p.69 |
| Chin | 200–400 Pa | MinInv p.69 |
| Marionette | < 200 Pa (Ch5) or 250–350 Pa (Ch7) — the book disagrees with itself | MinInv p.69, p.99 |
| Oral commissure | 250–350 Pa | MinInv p.99 |
| Upper lip lines | 150–220 Pa | MinInv p.99 |
| Radix, chin, deep grooves, sub-SMAS | high G' | ArtSci p.37, 39 |
| Anterior cheek | cohesive, softer | ArtSci p.37 |
| Periorbital, perioral superficial | flexible, low G', low water retention | ArtSci p.55, Periorb p.81 |
| Temple interfascial | medium (e.g. 338 Pa at 0.02 Hz) | MinInv p.111 |
| Forehead supraperiosteal, SOOF | medium G' | USArticle Fig 2, 8 |

Full 50-product rheology table at 0.02 Hz: MinInv Table 5.3, p.70–71 (reproduced in the extraction
notes). tan δ ≥ 0.25 means too fluid to hold shape (ArtSci p.40).

### 3.5 Detectability floor (A) — PRSFund p.603

Perceptive asymmetry thresholds: eyelid position 2 mm, oral commissure 3 mm, brow 3.5 mm, nasal tip
4 mm, chin 6 mm. Our pipeline noise: repeat-scan RMS ≈0.23 mm, volume floor ±0.12 mL (PLAN.md).
A per-region change is clinically visible only above the perceptive threshold; it is measurable
well below it.

### 3.6 Safety constants (B) — USArticle Table 1, ClinAnat p.175–176

Bolus < 0.2 mL per site, cannula 27 G or larger, retrograde threads, slow low-pressure injection.
Vascular occlusion incidence 1:6600; vision loss 577 cases 1996–2024, 40.6% nasal site, 79.6% HA.
Highest-risk regions for blindness: glabella, nose, nasolabial fold (ClinAnat p.173, PSCurr p.1115).

---

## 4. Region taxonomy

Sixteen regions, grouped into four behaviour classes. Boundaries follow AnatFFN p.31–33 (midface =
canthal line down to tragus–commissure line; anterior vs lateral midface split at lateral-orbital-rim
to commissure line; lower face = tragus–commissure line to mandibular border), the ArtSci Table 3.1
sub-SMAS space map (p.74), and the retaining-ligament anchors in AnatFFN p.57–61.

| Class | Behaviour | Regions |
|---|---|---|
| **P** bone-backed projection | supraperiosteal, wide smooth spread, high surface yield, long persistence | forehead, chin, jawline/prejowl, mandibular angle, pyriform/canine fossa, zygomatic/prezygomatic |
| **D** deep-space absorption | volume partly disappears into a compliant deep space, low surface yield per mL | temple, buccal/submalar, deep medial cheek |
| **T** thin-skin, low volume | small volumes, oedema-dominated early, high visibility per mL, Tyndall/lump risk | tear trough, palpebromalar, upper-lid sulcus, brow/ROOF, pretarsal |
| **M** mobile/muscular | migration, faster loss, superficial SAT placement gives localized projection | lips, perioral lines, marionette/commissure, nasolabial crease (superficial), glabella lines |
| **X** danger overlay | vascular-risk constraint that changes technique regardless of class | glabella, nose, nasolabial fold, temple |

---

## 5. Per-region rule sheets

Format: boundaries · layers · plane · volume · stiffness · spread/yield · timing · persistence ·
mobility · danger · variation. Grade in brackets.

### 5.1 Forehead (class P, overlay X at glabella)

- Boundaries: trichion to brows, laterally the temporal fusion line (AnatFFN p.7–8). Three superficial
  compartments: central, middle, lateral temporal; no deep fat (ArtSci p.131, PRSFund p.704).
- Layers: skin 1.70, fat 1.99 mm; total to bone 4.7 ± 0.3 mm; frontalis 2.0–3.3 mm deep [A]
  (ClinAnat p.6, 10). 8-layer stack incl. retrofrontalis fat (PRSFund p.484).
- Plane: supraperiosteal / subgaleal, "strongly recommended" [C] (ClinAnat p.102, ArtSci p.131,
  MinInv p.15). Cannula 23–25 G, 2–3 entries at brow, boluses connected by massage (ArtSci p.131).
- Volume: none stated [none].
- Stiffness: medium G'; firmer preferred because low-viscosity gel migrates under frontalis
  [B] (ArtSci p.131, USArticle Fig 2).
- Spread/yield: deep plane gives wide smooth spread [C]; expect high yield because only ~4.7 mm of
  tissue lies over bone. Mid-forehead soft frontalis "may swell" while the aponeurotic upper third
  stays flat, giving a step at the transition (ClinAnat p.106).
- Timing: oedema 2–3 days (ArtSci p.200); generic curve §3.2.
- Persistence: long (deep, bone-backed, low mobility) [C].
- Mobility: frontalis action moves soft filler; pre-treat hypertrophic corrugator with toxin
  (MinInv p.177).
- Danger: supraorbital artery splits 1 cm above rim (ArtSci p.66); supratrochlear superficial branch
  1.5 mm above rim (ArtSci p.130); supraorbital deep branch pierces frontalis 25–32.5 mm above rim
  (ClinAnat p.35); midline subcutaneous vessels in 52–60% (ClinAnat p.104); frontal branch STA crosses
  lateral frontalis 2 cm above lateral brow (ClinAnat p.103). Supraorbital nerve 27 mm and
  supratrochlear 17 mm from midline (Periorb p.190, PSCurr p.1150). Ultrasound mandatory (USArticle p.9).
- Variation: Asians brachycephalic, prefer convex forehead (ClinAnat p.100).

### 5.2 Glabella (class M for lines, P for depression; overlay X)

- Boundaries: between medial brows, frown lines 1–1.5 cm above frontal notch (ClinAnat p.102).
- Layers: skin "exceptionally thin"; supraorbital skin 1.67, fat 1.82 mm [A]; procerus 2.9 mm deep at
  sellion, 3.8 mm at glabella; corrugator 2–3 mm thick [A] (ClinAnat p.6, 12; ArtSci p.130).
- Plane: depression → supraperiosteal or subprocerus, >1 cm from rim; static lines → subdermal SAT
  small aliquots; fine lines intradermal [C] (ClinAnat p.102, 107, 109; ArtSci p.74, 80).
- Volume: none [none].
- Stiffness: high elastic modulus for radix/glabella under high external pressure (ArtSci p.37, 39).
- Spread/yield: SAT localized projection for lines; supraperiosteal spread for depression [C].
- Persistence: shorter for dermal placement [C].
- Mobility: corrugator/procerus; toxin first, filler for residual (MinInv p.177; Oculo p.207).
- Danger: highest blindness rate of any site (DermUS p.127, ArtSci p.130); ophthalmic artery
  emergence 14 mm above / 4 mm medial to medial canthus, 5 mm deep (ClinAnat p.35); intercanthal
  vein subcutaneous, 71–100% (ClinAnat p.40, 154). Cannula "highly recommended"; ultrasound mandatory.

### 5.3 Brow and upper-lid sulcus (class T)

- Boundaries: brow from supraorbital rim line to the fold; sulcus = preseptal space below the orbital
  retaining ligament, lateral to mid-pupil for brow (ArtSci p.141, 138).
- Layers: eyelid skin 0.38–0.80 mm, no subcutaneous fat [A]; ROOF deep to orbicularis, atrophies
  with age (ArtSci p.141, Periorb p.5).
- Plane: brow → submuscular into ROOF, moderate HA, then soft subdermal; sulcus → preseptal, under
  orbicularis, above septum, patient upright, under-correct [C] (ArtSci p.141, 138–139; ClinAnat p.111).
- Volume: fat 1–3 mL per brow, 1–2 mL per sulcus [B] (Periorb p.91–92); HA none stated.
- Stiffness: soft moldable HA (ArtSci p.137).
- Spread/yield: high visibility per mL; over-correction → puffy or inferior migration into lid
  (ArtSci p.138, ClinAnat p.111).
- Danger: supraorbital/supratrochlear anastomose with orbital arteries; contraindicated in ptosis
  or proptosis (ArtSci p.138, 140).
- Variation: brow ptosis less severe in Asians; brow filler common in West, not Korea (ArtSci p.7, 140).
  Brow falls ≈2.5 mm per 20 years (PRSFund p.607).

### 5.4 Temple (class D, overlay X)

- Boundaries: temporal fossa lateral to the fusion line; typical hollow in the lower temple just above
  the arch (ArtSci p.77). Upper vs lower temporal compartment split by the inferior temporal septum.
- Layers: skin 1.65, fat 2.58 mm [A] (ClinAnat p.6); 10-layer ultrasound stack with STA in layers
  2–4, middle temporal vein layers 5–7, deep temporal artery on bone [A] (USProto p.68–70);
  DTF splits ≈2.5 cm above the arch (AnatFFN p.16); skin-to-bone "quite variable" (ClinAnat p.121),
  28 mm in one patient (USArticle).
- Plane: four options (ClinAnat Fig 3.33, p.118; ArtSci Fig 5.7, p.136). Interfascial (between STF
  and DTF) is the authors' choice in ArtSci and MinInv; supraperiosteal deep to temporalis is
  ClinAnat's choice but "may not show change despite a large amount" (ArtSci p.136) and "needs larger
  volume" (ClinAnat p.120) [C].
- Volume: fat 2–3 mL per temple [B] (Periorb p.92); HA none stated.
- Stiffness: medium interfascial (338 Pa at 0.02 Hz), high viscosity deep with needle (MinInv p.111,
  ArtSci p.208).
- Spread/yield: **lowest surface yield of any region when injected deep** [C]; temporalis contraction
  moves deep filler (ArtSci p.77). Undulations common in thin skin superficially (ClinAnat p.121).
- Persistence: long in deep planes [C].
- Danger: STA bifurcation 37 mm above / 18 mm anterior to tragus (ClinAnat p.40, MinInv p.3); middle
  temporal vein ≈5 mm diameter, 2 cm above arch (ArtSci p.136), 0.5–9.1 mm (MinInv p.6); sentinel vein
  2 mm; frontal branch VII 1.5–3 cm above arch (ArtSci p.135). Ultrasound mandatory (USArticle p.9).
- Variation: hollowing worse with low BMI (ArtSci p.134); bi-temporal width decreases with age
  (PRSFund p.606).

### 5.5 Tear trough and nasojugal groove (class T)

- Boundaries: medial canthus to mid-pupillary line, ≈3 cm long, 2–3 mm below the rim, along the
  tear-trough ligament (medial ORL) [A] (ArtSci p.147). Nasojugal fold runs inferolaterally at 45°
  (AnatFFN p.33). Medial-most trough has no SOOF: "skin, muscle and bone" only (MinInv p.51,
  ArtSci p.81, 151).
- Layers: infraorbital skin 1.97, fat 4.93 mm just below the rim [A] (ClinAnat p.6); lid skin
  0.37 to >2 mm, no fat; orbicularis 1.56 mm at 0.88–2.8 mm [A] (MinInv p.41, 177).
- Plane: supraperiosteal / sub-orbicularis along the rim, "the only area where it is advisable to
  inject deep under the muscular plane" (Periorb p.82); two-layer principle (SOOF deep + thin subdermal
  finish) is the Korean consensus (ClinAnat p.126, 129; ArtSci p.152) [C].
- Volume [B, best-documented region]: ≤0.5 mL per trough, published mean 0.26 mL (range
  0.08–0.32) (MinInv p.51); 0.2–0.3 mL per eyelid (Periorb p.82); MD Codes Tt1/Tt2/Tt3 = 0.2/0.2/0.1
  mL per side, may be doubled (Periorb p.82); 0.1–0.2 mL per pass; overall infraorbital 0.2–1 mL
  (MinInv p.58); micro-boluses 0.3–0.5 mL, total 0.3–1 mL (PSCurr p.1114); intraoral nasojugal
  0.3–0.5 mL (ClinAnat p.136); fat 2–3 mL (Periorb p.92).
- Stiffness: lowest G', minimal water retention; Volbella 99 Pa, Belotero Balance 21.7 Pa, Restylane
  349 Pa at 0.02 Hz (MinInv p.49, 70). No CaHA or collagen (ClinAnat p.126).
- Spread/yield: highest visibility per mL of any region (thin skin, bone close); tunnel wider than
  the target so filler spreads laterally rather than bulging (ArtSci p.153).
- Timing: high-water HA swells 1–2 weeks (ClinAnat p.126); saturation ≈2 months, follow-up at
  8 weeks (MinInv p.141); HA "expand[s] as they imbibe water overnight" → undertreat (Oculo p.206).
  Delayed complications mean onset 16.8 months: swelling 42.3%, nodules 25% (MinInv p.56).
- Persistence: >12 months (MinInv p.141).
- Mobility: orbicularis pushes superficial filler; supraperiosteal HA can sit inside muscle and be
  displaced superficially → oedema (MinInv p.57). Filler above the malar septum → chronic oedema.
- Danger: detoured infraorbital branch of the facial artery along the groove in ≈30% of Asians
  (ClinAnat p.129, 135; ArtSci p.154; MinInv p.48); infraorbital foramen 6–10 mm below rim (MinInv p.47),
  2–3 cm lateral of midline (Periorb p.201); infraorbital artery 3.1 mm deep in one patient (USArticle).
- Variation: Asians thicker skin, more fat; haemosiderin risk Fitzpatrick 4–6 (MinInv p.42, 56);
  lid-cheek junction lower in men (Periorb p.16).

### 5.6 Palpebromalar groove and midcheek groove (class T → P)

- Boundaries: mid-pupil to lateral canthus along the ORL (looser, longer, easier to fill); midcheek
  "Indian band" along the zygomatico-cutaneous ligament (ArtSci p.147–151, 156). Orbitomalar ligament
  from 5 mm lateral of the lateral rim to the anterior lacrimal crest, strongest laterally (AnatFFN p.58).
- Plane: small boluses in front of the septum, supraperiosteal between ORL and zygomatic-cutaneous
  ligament, entry 3 cm below lateral canthus (MinInv p.52); midcheek groove: tunnel the ligament
  (~10 passes) then medium filler, or deep medium-high supraperiosteal towering beneath the ligament
  (ArtSci p.156–157) [C].
- Volume: ≤1 mL, 0.5–1 mL first visit; volume matters more than brand for swelling (MinInv p.52).
- Yield/spread: filler above the midcheek crease makes it look deeper (ArtSci p.72).

### 5.7 Zygomatic / anterior malar (prezygomatic space) (class P)

- Boundaries: apple-cheek apex at intersection of alar-groove→helix-root line and lateral-canthus→
  commissure line (Asian) or above/lateral to the cheekbone (Caucasian) (ArtSci p.157). Zygomatic
  ligament ≈45 mm anterior to tragus, width 14.5 mm, elongates only 9 mm with age (AnatFFN p.58–59,
  PRSFund p.595). Prezygomatic space: 7 layers, roof orbicularis, floor zygomaticus origins, no
  important neurovascular structures (ArtSci Table 3.3, p.82).
- Layers: cheek skin 1.85, fat 4.54 mm [A]; skin >2 mm at zygion (ClinAnat p.5–6).
- Plane: supraperiosteal / prezygomatic bolus, needle to bone or cannula [C] (ArtSci p.158, PSCurr
  p.1119, MinInv p.15 in 0.05–0.1 mL aliquots). High G' oval pockets on periosteum (DermUS p.526).
- Volume: case 3 mL HA cheeks+temples (PRSFund p.615); fat 2–3 mL per deep compartment (PSCurr
  p.1195) [B].
- Stiffness: high G' supraperiosteal; cohesive softer for anterior cheek (ArtSci p.37, USArticle Fig 8).
- Spread/yield: SSRT lifting rule: anterior malar fill lifts the nasolabial area, side cheek lifts the
  jawline (ArtSci p.89). Malar volumisation "may decrease the need to treat the tear trough"
  (PSCurr p.1115).
- Persistence: "persist for years" by MRI (MinInv p.148).
- Danger: transverse facial artery perforator in McGregor's patch (AnatFFN p.61); zygomaticofacial
  foramen 2 cm lateral + 2 cm inferior to lateral canthus (Oculo p.22). Ultrasound recommended.
- Variation: Koreans show no age-related midface bone loss, hollowing is soft-tissue ptosis →
  augment the soft-tissue plane, not bone (ArtSci p.6, 81, 158–159).

### 5.8 Deep medial cheek / midcheek hollow (class D)

- Boundaries: anterior midface triangle between lower lid and nasolabial fold (AnatFFN p.32);
  premaxillary space between zygomatic-cutaneous and buccal-maxillary ligaments, medial boundary the
  angular artery, lateral the angular vein (ArtSci p.82).
- Layers: same as 5.7; deep medial cheek fat below SMAS, small adipocytes, shrinks with age
  (ArtSci p.69, MinInv p.9–10).
- Plane: deep cheek fat superficial to the deep facial muscles (under SMAS, above zygomaticus),
  spread evenly, no boluses so it moves with the smile (ArtSci p.158) [C].
- Volume: fat 2–3 mL per deep compartment (PSCurr p.1195); HA none stated.
- Yield: moderate, deep fat pad → oval pocket that stays for months (DermUS p.526).
- Danger: facial vein along the nasojugal groove, deeper than deep fat near the NLF (ArtSci p.158).

### 5.9 Lateral cheek, submalar, buccal hollow (class D)

- Boundaries: lateral midface over the arch and below it; hollow bounded by the arch above, mid-masseter
  below, masseter borders (ClinAnat p.142–143). Masseteric ligaments 42 mm anterior to tragus back to
  25 mm posterior of the masseter edge (AnatFFN p.59). Buccal fat ≈10 mL, grows to ~50 y then shrinks
  (ArtSci p.70, 85); its intermediate lobe is "the target of cosmetic fillers" (AnatFFN p.74).
- Layers: SMAS thickest preauricular, thinning medially (ArtSci Table 3.2); subcutaneous fat over the
  parotid "notoriously thin" (DermUS p.528).
- Plane: subcutaneous fat over masseter for mild; preparotid-premasseteric sub-SMAS space after partial
  ligament tunnelling for severe; prebuccal space rather than into the buccal fat (ArtSci p.83–86,
  161–163; ClinAnat p.139–143) [C].
- Volume: none [none]; "requires a large volume" if into buccal fat (ClinAnat p.139).
- Yield: low; dense connective tissue makes volume increase "difficult", massive filler → undulations
  on animation (ClinAnat p.143). Deep-to-masseter HA meanders between fibres (DermUS p.526).
- Danger: parotid duct 2 cm below the arch, 2–3 mm diameter (AnatFFN p.65); transverse facial artery
  loops (USArticle Fig 6); buccal branch between premasseteric spaces (ArtSci p.72).
- Variation: Asians need fill below the arch, not on it (ArtSci p.7).

### 5.10 Nose (class P thin, overlay X)

- Boundaries: radix at the intercanthal line (Asian) to tip and columella. 5 layers: skin, superficial
  fat, fibromuscular, deep fat, perichondrium (ClinAnat p.146, PRSFund p.648).
- Layers: radix/dorsum skin 1.51, fat 1.61 mm, thinnest fat on the face [A] (ClinAnat p.4, 6);
  tip skin ≈2 mm, thick dermis, no soft fat (ArtSci p.168). Fibroadipose, not lobular, hypodermis
  (DermUS p.113).
- Plane: supraperiosteal / supraperichondrial only, cannula ≥25 G from tip, radix relocated 3–5 mm
  superior; columella deep subcutaneous 0.2–0.5 mL [B/C] (ClinAnat p.146–153, ArtSci p.168–170).
- Volume: columella 0.2–0.5 mL (ClinAnat p.153); tip "small amount, droplet" [B].
- Stiffness: high G', low swelling; monophasic widens the dorsum (ClinAnat p.149, ArtSci p.166–168).
- Yield: high (bone/cartilage backed); radix under high external pressure sinks within weeks if the
  gel is too soft (ArtSci p.56, 122).
- Danger: dorsal nasal artery superficial over the bony dorsum, deep over the cartilaginous dorsum
  (ClinAnat p.36, 145); nose = 40.6% of vision-loss cases (USArticle p.2); highest compression-necrosis
  risk (ArtSci p.204); post-rhinoplasty arteries fixed by scar (USArticle p.9). Ultrasound mandatory.

### 5.11 Pyriform / canine fossa and nasolabial fold (class P deep, M superficial; overlay X)

- Boundaries: alar-facial crease to the commissure; Ristow's space = canine fossa under medial deep
  medial cheek fat (ArtSci p.73, 82). Facial artery 3.2 ± 4.5 mm lateral to the ala, 13.5 ± 5.4 mm
  from cheilion, within 5 mm of the fold in 43%, medial to the fold in >70% of Koreans [A]
  (ClinAnat p.133–135, ArtSci p.176).
- Layers: superficial fat thickest lateral to the fold (5.14 ± 3.31 mm) with an abrupt change across
  it [A] (ClinAnat p.4, 6).
- Plane: two layers: firm filler in Ristow's space / canine fossa first, then soft filler subdermal
  along the crease [C] (ArtSci p.174–176, ClinAnat p.136, MinInv p.110). Deep filler migrates above the
  crease unless blocked with a finger (ArtSci p.82).
- Volume: pyriform bolus 0.1–0.2 mL (PSCurr p.1118) [B]; crease none stated.
- Stiffness: firm deep, soft superficial; mouth-corner fillers must not be diluted with free HA
  (ArtSci p.49).
- Spread/yield: large subcutaneous volumes → lateral expansion not projection; deep → projection
  (ArtSci p.176). SAT crease fill → localized projection (ClinAnat p.53).
- Mobility: deep NLF filler can migrate superolaterally with expression and deepen the fold
  (ArtSci p.59).
- Danger: exposed facial artery segment ≈15 mm lateral to cheilion (ClinAnat p.37); superior labial
  artery branches within a 1.5 cm square superolateral to cheilion, minimum 3 mm deep (ClinAnat p.138);
  facial artery loops in the groove (USArticle Fig 3). Blindness-risk region.

### 5.12 Lips (class M)

- Boundaries: vermilion, philtrum, commissures; ideal upper:lower 1:1.6 (ArtSci p.9).
- Layers: lip thickness at the vermilion border upper 9.4 ± 0.4, lower 10.9 ± 0.7 mm [A]
  (ClinAnat p.158); no subcutaneous fat (DermUS p.90); labial arteries in the wet-mucosal layer,
  deeper than orbicularis, in 28–57% (ClinAnat p.158); superior labial artery ≈1 mm diameter, 0.5–1.2
  cm above the border (ArtSci p.179, MinInv p.5).
- Plane: submucosal / superficial fat at the dry–wet junction for volume; dermal/subdermal along the
  border for definition [C] (ClinAnat p.158–160, ArtSci p.180).
- Volume [B]: 0.3–0.4 mL per lip typical, tubercle 0.1–0.2 mL, max 1.5 mL per lip (ArtSci p.180–181);
  upper-lip lines ≈1 mL per session (MinInv p.99).
- Stiffness: 150–220 Pa for lip lines (MinInv p.99); HA only, never CaHA (ClinAnat p.158, PSCurr p.1115).
- Timing: **10–20% of volume lost at 1 week** (ArtSci p.181) [B].
- Persistence: shorter, "very dynamic", repeat sessions (ClinAnat p.169).
- Yield: moderate; beyond 1.5 mL the lip thickens rather than rotates (ArtSci p.181).
- Danger: persistent-caliber labial artery variant in the dermis (DermUS p.127, USArticle Fig 4).
- Variation: superficial lip fat increases with age, deep decreases, total conserved (ArtSci p.179);
  Asians want the central tubercle (ArtSci p.181).

### 5.13 Marionette and oral commissure (class M)

- Boundaries: modiolus to mandibular border along the DAO borders; modiolus 11.0 ± 2.6 mm lateral,
  8.9 ± 2.8 mm below cheilion in Asians [A] (ClinAnat p.17). Marionette is medial to the facial artery
  → low occlusion risk (MinInv p.104).
- Layers: subcutaneous thickness changes abruptly at the line (ClinAnat p.4); skin thicker than the
  upper lip (MinInv p.99).
- Plane: subdermal release then subcutaneous above muscle for mild; cannula into sub-DAO fat for deep
  hollows; cross-hatching across layers [C] (ClinAnat p.169, ArtSci p.185, MinInv p.99).
- Volume [B]: marionette 0.5–1 mL, commissure 0.2–0.4 mL per side (MinInv p.99); "minimum volume"
  for jowl, excess aggravates sagging (ClinAnat p.171).
- Stiffness: 250–350 Pa or <200 Pa (MinInv, contradictory).
- Yield: moderate; do not put volume outside the line (adds heaviness) (MinInv p.99).
- Danger: inferior labial, mental artery at the foramen 2 cm below the commissure (ClinAnat p.45).

### 5.14 Chin (class P)

- Boundaries: labiomental crease to menton, laterally to the mandibular ligament / prejowl.
  Labiomental depth ≈4 mm women, 6 mm men (PRSFund p.728). Filler suitable for chin–lip deficit
  <7 mm; osteotomy >10 mm (PRSFund p.730).
- Layers: skin ≈2 mm, >2 mm at pogonion [A]; mentalis 6.7–10.6 mm deep [A] (ClinAnat p.5, 83); thin
  deep fat under mentalis at the tip (ArtSci p.190).
- Plane: supraperiosteal bolus beneath mentalis, needle to bone, then medium superficial top-up
  [C] (ClinAnat p.164–165, ArtSci p.191, MinInv p.73, PSCurr p.1118).
- Volume [B]: 2–4 mL of the hardest filler (ClinAnat p.167); 0.2–0.5 mL per side deep + 0.1 mL
  subcutaneous (MinInv p.73). The two sources differ by an order of magnitude; ClinAnat is a
  full-projection Korean protocol, MinInv a refinement dose.
- Stiffness: 200–400 Pa (MinInv p.69); high G' / CaHA (ArtSci p.191).
- Yield: high (bone-backed). Oval pocket on periosteum (DermUS p.526). Tissue tension limits volume
  per session, split over 2–3 sessions (ClinAnat p.165, 167).
- Timing: contours exaggerated immediately, normalise 1–2 weeks, final 2–4 weeks (MinInv p.77).
- Mobility: **hyperactive mentalis migrates filler superiorly and shortens duration**; toxin 2–4 U per
  side prolongs shape (ClinAnat p.165–167, ArtSci p.192, MinInv p.179). Migrates toward the mouth or
  down the platysma plane if placed too high or too low (ArtSci p.194).
- Danger: mental foramen; median perforating mandibular artery variant → tongue necrosis after
  supraperiosteal chin bolus (USArticle Fig 5).
- Variation: Asian lower-face ratio 1:1:0.8, prefer forward-and-down elongation; Western prefer a
  distinct angle (ClinAnat p.167, ArtSci p.7, 189). Chin asymmetry perceptible only at 6 mm (PRSFund p.603).

### 5.15 Jawline, prejowl and mandibular angle (class P)

- Boundaries: a band ≈1.5 cm wide parallel to the inferior border from prejowl to angle (MinInv p.69).
  Mandibular ligament at the prejowl sulcus anterior border (ClinAnat p.4); mandibular septum separates
  superior and inferior jowl fat (USProto p.31).
- Layers: subcutaneous fat thinner than the chin (ArtSci p.192); masseter region skin-to-bone
  15–25 mm, masseter 14.9 ± 2.2 mm thick on 3D-CT [A] (ClinAnat p.88).
- Plane: supraperiosteal for structure; subdermal over platysma is "devoid of major neurovascular
  structures" (MinInv p.65, 69); prejowl combined deep + subcutaneous (MinInv p.73) [C].
- Volume: case 3 mL CaHA jawline + marionette (PRSFund p.615); TSST 0.02 mL aliquots at the angle
  (MinInv p.15) [B].
- Stiffness: >400 Pa (MinInv p.69).
- Yield: high over bone; low over masseter (meandering intramuscular spread, DermUS p.526).
- Timing: swelling normalises 1–2 weeks (MinInv p.77).
- Danger: facial artery at the antegonial notch covered only by skin and platysma (AnatFFN p.68);
  marginal mandibular nerve crosses the artery ≈3 cm anterior to the gonion, 4 cm safety rule
  (AnatFFN p.81), or 22–23 mm anterior / 30–31 mm above the angle (PRSFund p.720); parotid drift
  (MinInv p.115).
- Variation: Asians augment the chin only and avoid angle fill (ArtSci p.7).

### 5.16 Neck and preauricular (none)

No filler volume, plane, or response data in any book. Neck: platysma decussation 85% Asian vs 39%
Caucasian non-decussation (ClinAnat p.21), three fat compartments (PRSFund p.721). Preauricular:
SMAS thickest, little subcutaneous fat (AnatFFN p.82), earlobe 0.5–1 mL per side (ClinAnat p.172).
Treat both as measurement-only regions with no response prior.

---

## 6. What is a prior and what must be fitted

| Rule field | Source in books | Status |
|---|---|---|
| Region boundaries | AnatFFN, ArtSci Table 3.1, ligament distances | **Prior (C/A)**, implement as landmark polygons |
| Skin + superficial fat thickness | ClinAnat Table 1.1 | **Prior (A)**, Korean population; expect thicker in Caucasians per ArtSci p.6 note on Asians being thicker (contradictory across books, treat as ±30%) |
| Injection plane per region | all filler books agree | **Prior (C)** |
| Typical volume per region | tear trough, lips, chin, marionette, columella, pyriform | **Prior (B)** for those six; none elsewhere |
| Stiffness band | MinInv, ArtSci | **Prior (B)** |
| Spread pattern (footprint vs peak) | SAT/DAT rule | **Prior (C)**, qualitative only: needs footprint radius per region from our scans |
| Surface yield (surface ΔV / injected V) | only ordinal hints (temple deep low, forehead high, lips 0.8–0.9 at 1 wk) | **Must fit** |
| Time curve | ArtSci generic curve; region modifiers for tear trough, lips, chin | **Prior (B)** for shape; region scaling must fit |
| Persistence | ordinal by plane and mobility | **Prior (C)** ordinal; half-lives must fit |
| Migration direction | chin up, NLF superolateral, malar caudal (PAAG), infraorbital superficial | **Prior (C)**, verify |
| Danger zones | all books | **Prior (A/B)**, display-only for the scanner |
| Detectability floor | PRSFund thresholds + our noise floor | **Prior (A)** |

Fitting plan for the four "must fit" fields, using the existing comparison pipeline:

1. Tag every real before/after pair with region, plane, product, injected mL, and days since injection.
2. Per region, regress measured ΔV against injected mL × the §3.2 time factor; the slope is the
   surface yield, the residual is the region's absorption into deep spaces.
3. Per region, fit the detected change footprint (equivalent radius) and peak displacement; expect
   class P/D to have large radius and low peak, class T/M small radius and high peak.
4. Re-scan the same subjects at 2, 8, 16 weeks to fit the region-specific time curve against the
   ArtSci shape.

The published thresholds give a sanity check: a 0.26 mL tear-trough dose over a ≈3 cm × 1 cm
footprint is ≈0.9 mm mean displacement, four times our 0.23 mm noise but below the 2 mm eyelid
perceptive threshold. A 3 mL cheek dose over a 4 cm × 4 cm footprint is ≈1.9 mm mean displacement.

---

## 7. Papers cited in the books that would fill the gaps

These are the primary sources the books lean on but never reproduce. They are the next data to
pull if the rule set needs measured numbers rather than expert ranges.

- Gierloff et al. 2012, CT study of midfacial fat-compartment ageing (cited AnatFFN p.85, MinInv p.12, 80) — the only quantitative compartment-volume-loss source named.
- Cotofana et al. 2021, plane change of supratrochlear/supraorbital arteries in the forehead, ultrasound (USArticle ref 24).
- Phumyoo et al. 2020, Clin Anat 33:370, central forehead artery localisation (USArticle ref 22).
- Velthuis et al. 2021, Aesthet Surg J 41:NP1621, standard Doppler positions per region (USArticle ref 14).
- Sigrist et al. 2024, Diagnostics 14:1718, ultrasound guidance, upper third (USArticle ref 17).
- Wu et al. 2025, J Cosmet Dermatol 24:e70164, lip sonoanatomy (USArticle ref 29).
- Gonzalez et al. 2025, Cureus 17:e79325, ultrasound of cannula vs needle lip filler spread (USArticle ref 16) — the closest thing to a measured spread-per-plane study.
- Khorasanizadeh et al. 2023, J Cosmet Dermatol 22:1844, Doppler survey of facial artery variants (USArticle ref 35).
- Lee W et al. 2020, Dermatol Surg, rheology of 50 HA fillers (MinInv Table 5.3 source).
- Bernardini 2025, tear-trough mean volume 0.26 mL (MinInv p.51).
- Master 2024, review of 33 MRI studies showing midface HA persists for years (MinInv p.148).
- Marten & Elyassnia 2018, "Facial fat grafting: why, where, how, and how much" (PSCurr bibliography) — per-region fat volumes.
- Rohrich 2014 lift-and-fill; Ramanadham & Rohrich 2015 fat compartments; Cotofana 2015 midface; Trévidic 2022 "smiling cadavers" (PSCurr bibliography).
- Liu et al. 2023, age-related periocular morphology, 2D and 3D anthropometry in Caucasians (Periorb ref 25) — the only 3D-anthropometry study cited anywhere.
- Ambroziak et al. 2019, elastography reference values of facial skin (USProto refs).

---

## 8. Caveats

- Nine of the eleven mined sources are either Korean-authored or ophthalmology/board-review texts.
  Thickness numbers and vessel positions are Korean cadaver data; the books themselves flag that
  Caucasians lose midface bone where Koreans do not (ArtSci p.6, 81) and that facial artery
  continuation to the angular artery is 36% in Asians vs 4–68% elsewhere (ClinAnat p.37).
- Volume guidance is per-session clinical habit, not outcome data.
- The ArtSci time curve is presented without a study citation in the extracted text; treat the 1.8×
  peak as a shape prior, not a constant.
- MinInv gives two different marionette G' bands in two chapters; both are recorded.
- None of the books measure surface displacement, so every "yield" statement above is ordinal.
