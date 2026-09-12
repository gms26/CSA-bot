# AppleSupport Intent Taxonomy

## I1 — Software / iOS / macOS Issue

Use when the primary problem is a system-level software or operating-system
problem.

Examples:
- iOS/macOS bugs
- system-wide freezing
- OS performance problems
- update failures
- unexpected system behavior
- crashes/restarts caused by the OS
- keyboard/autocorrect/UI bugs

Rule:
If the user explicitly attributes multiple or broad problems to an OS/software
update, use I1.

---

## I2 — Battery / Charging / Power

Use when battery, charging, overheating related to charging, or power is the
main problem.

Examples:
- battery draining quickly
- poor battery life
- charging problems
- slow charging
- phone dying unexpectedly

Rule:
A battery-specific problem gets I2 even when the user says an iOS update
caused it.

---

## I3 — App / App Store / App Update Issue

Use when the primary problem is with a specific app, the App Store, or app
installation/update.

Examples:
- an app crashes
- an app does not open
- app functionality is broken
- App Store is unavailable
- apps cannot update
- apps cannot be installed

Rule:
A specific application problem is I3 unless the problem is clearly a
system-wide OS problem.

---

## I4 — Connectivity / Network

Use for network and connection problems.

Examples:
- Wi-Fi
- cellular/mobile data
- Bluetooth
- hotspot
- websites/network access
- Apple Watch cellular connection

Rule:
If a specific connectivity function is failing, use I4 even when an OS update
is blamed as the cause.

---

## I5 — Messages / Communication

Use for messaging and communication functionality.

Examples:
- iMessage
- SMS/messages
- message delivery
- messages disappearing
- email sending/receiving problems

---

## I6 — Device / Hardware Issue

Use when the problem is primarily physical-device hardware or hardware
functionality.

Examples:
- broken/unresponsive screen
- Touch ID hardware behavior
- device won't turn on
- defective headphones
- replacement screen problems

Rule:
If the message clearly indicates a physical/hardware problem, use I6.

---

## I7 — Account / iCloud / Security

Use for account access, iCloud account issues, security, or suspected
account compromise.

Examples:
- Apple ID/account problems
- iCloud account problems
- password/account access
- suspicious account activity
- account locked
- data/account recovery when clearly tied to account or iCloud access

---

## I8 — Apple Services / Media

Use for Apple services and media-related functionality.

Examples:
- Apple Music
- Podcasts
- purchased music
- Photos/iCloud Photos
- Apple media services
- service subscription access when the core issue is service availability

Rule:
Use I9 instead when the main problem is a purchase, refund, billing,
reservation, repair, replacement, or warranty transaction.

---

## I9 — Purchase / Order / Repair / Warranty

Use for commercial, transaction, repair, replacement, and warranty-related
requests.

Examples:
- iPhone preorder
- reservation/delivery
- refund
- upgrade program
- repair
- replacement
- warranty
- purchase-related billing
- gift vouchers

---

## I10 — How-to / Settings / Information

Use when the customer primarily asks how to perform an action, change a
setting, or obtain information.

Examples:
- how to change a setting
- how to delete an app
- how to check battery health
- how to pair devices
- how to configure APN
- how to free storage

---

## I11 — Other / Unclear

Use when the message cannot be reliably assigned to another intent.

Examples:
- "Yes"
- "Thanks"
- "Sure"
- "11.0.3"
- "I already tried that"
- generic complaints without a specific problem
- messages that depend entirely on missing conversation context
- image/link-only messages with no useful description

Rule:
Do not guess an intent when the message does not provide enough information.