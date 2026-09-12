# AppleSupport Intent Taxonomy

## I1 — Software / iOS / macOS Issue

### Include
- iOS/macOS update problems
- freezing
- crashes caused by software
- performance degradation
- software glitches

### Exclude
- battery-specific problems
- app-specific problems
- network-specific problems

---

## I2 — Battery / Charging / Power

### Include
- battery drain
- charging problems
- slow charging
- power-related issues

### Exclude
- general performance problems
- hardware damage unrelated to power

---

## I3 — App / App Store Issue

### Include
- app crashes
- app installation
- app updates
- App Store problems

---

## I4 — Connectivity / Network

### Include
- Wi-Fi
- cellular data
- signal
- Bluetooth
- hotspot

---

## I5 — Messages / Communication

### Include
- iMessage
- SMS/messages
- message delivery
- message activation problems

---

## I6 — Device / Hardware Issue

### Include
- screen
- camera
- Touch ID
- physical device problems
- device not powering on

---

## I7 — Apple Account / iCloud / Security

### Include
- Apple ID
- iCloud
- password
- account access
- suspicious account activity

---

## I8 — Media / Apple Services

### Include
- Apple Music
- purchased music
- podcasts
- other Apple media/service problems

---

## I9 — Purchase / Order / Repair / Warranty

### Include
- preorder
- orders
- returns
- replacement
- AppleCare
- warranty
- refunds

---

## I10 — How-to / Settings / Information

### Include
- configuration questions
- settings
- APN
- battery health
- storage management
- "How do I..." questions

---

## I11 — Other / Unclear

### Include
- insufficient information
- vague complaints
- acknowledgements
- "Thanks"
- "Yes"
- "DM sent"
# Labeling Rules

## Primary-intent rule

Each customer message receives exactly ONE primary intent.

Choose the intent that best represents the customer's main problem or request.

Do not assign multiple intents.

## Multi-symptom rule

If a message contains several symptoms, choose the main problem.

Example:

"After iOS 11 my phone is slow, freezing and apps are crashing."

Label:
I1 - Software / iOS / macOS Issue

Reason:
The customer attributes the symptoms to the operating-system update.

## Specific-over-general rule

When a clearly identifiable specific problem exists, prefer the specific intent.

Examples:

"Apple Music won't open"
→ I8 - Media / Apple Services

"None of my apps will update"
→ I3 - App / App Store Issue

"My iPhone has no cellular signal"
→ I4 - Connectivity / Network

## How-to rule

Questions asking how to configure, check, enable, disable, or find a setting belong to:

I10 - How-to / Settings / Information

Example:

"How can I check my iPhone battery health?"
→ I10

But:

"My battery drops from 100% to 20% in an hour"
→ I2

## Hardware vs software rule

Use I6 when the problem appears to be physical/device hardware.

Use I1 when the complaint is primarily about software behavior.

Example:

"Touch screen is physically unresponsive"
→ I6

"Phone freezes after iOS update"
→ I1

## Other / unclear rule

Use I11 when there is not enough information to determine a meaningful intent.

Examples:

"Yes"
"Thanks!"
"DM sent"
"Fix this!"
## Additional Boundary Rules

### Explicit software/update attribution

When a customer mentions multiple symptoms but explicitly attributes them
to an iOS/macOS update or software change, classify the primary intent as:

I1 - Software / iOS / macOS Issue

Example:
"After the iOS update my phone is slow and apps are crashing."
→ I1

### Insufficient context

If a message is a continuation of a conversation and cannot be
understood independently, do not guess the intent.

Use:
I11 - Other / Unclear

Example:
"Yes."
→ I11

Example:
"I already tried that."
→ I11

Example:
"It was available."
→ I11

### Complaint without a concrete problem

General dissatisfaction with Apple or customer service, without a
specific identifiable problem, belongs to:

I11 - Other / Unclear