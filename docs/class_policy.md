# Which classes we emit, and why

Score A is macro-F1 over the classes that appear in the test set **or in our
predictions**. A class we predict that never occurs adds a 0 to the average.
So a class is only worth emitting when its rule is specific enough to be right
most of the time. Every decision below was made by inspecting the rendered
frames of every candidate event on the four sample videos
(`scripts/replay_rules.py` + frame montages). We have no labelled dev set yet
(see README → Limitations), so these are judgement calls, not measured F1.

| class | status | rule (see `src/rules.py`) | what the sample spot-checks showed |
|---|---|---|---|
| `jaywalking` | on | pedestrian (not a rider or passenger) ≥ 0.1 diag inside the carriageway and ≥ 0.4 diag away from any zebra, island, divider or sidewalk, for ≥ 1.5 s | Mostly people crossing the junction diagonally or crossing the far carriageway mid-block. Some misses of people walking just off a zebra, which we trade for precision |
| `failure_to_yield` | on | vehicle moving (≥ 0.4 diag/s) through a zebra while a pedestrian who has been ≥ 0.25 diag inside the *same* zebra for ≥ 0.5 s is ahead of it within its swept width | After requiring "same zebra, ahead, inside (not at the kerb)" and excluding riders, the remaining events show cars turning across occupied crossings. The first version fired on every car passing a waiting pedestrian |
| `stop_line` | on | on red, a vehicle that came from `lane_ltr` stands (≤ 0.08 diag/s) on the first zebra, i.e. past both the stop line and the queue-tolerance line, without entering the junction | Cars stopped on the zebra during red |
| `red_light` | on | a `lane_ltr` vehicle crosses the stop line forwards while the signal has been red ≥ 1 s, then drives into the junction within 5 s; crossings ≤ 2 s before green count as anticipating the green | Rare (1 on the samples). The signal itself is reliable on all four videos (`scripts/signal_timeline.py`) |
| `stopped_vehicle` | on | a vehicle seen moving at some point, now stationary ≥ 10 s on the carriageway, not at a zebra, not in the junction box, not queued behind another stopped vehicle, not a `lane_ltr` signal queue (moves off ≤ 15 s after green) | Rare. Signal queues, parked cars and vehicles yielding at zebras are removed |
| `congestion` | on | ≥ 4 moving-lane vehicles in one carriageway with median speed < 0.12 diag/s for ≥ 8 s; `lane_ltr` must persist ≥ 12 s into green (not a red queue) | Rare. Ordinary red queues and parked cars no longer count |
| `wrong_way` | on | vehicle in a carriageway (outside the junction) moving ≥ 0.4 diag/s with heading cos < −0.6 against the lane flow for ≥ 1.5 s | 0 events on the samples (none expected) |
| `near_miss` | on | two road users close (< 0.8 diag), closing ≥ 0.3 diag in the last second, one braking hard (speed < 40 % of the previous second's), no box overlap | 0 events on the samples |
| `accident` | on | anomaly model (conf ≥ 0.6) on a box that covers a vehicle, on the carriageway, in ≥ 3 of 4 consecutive 1 Hz checks | 0 events on the samples (the model fired on normal traffic at conf 0.45 in earlier versions) |
| `fire_smoke` | on | anomaly model fire/smoke (conf ≥ 0.6) on the carriageway in ≥ 3 of 4 checks | 0 events on the samples |
| `road_obstacle` | on | animal / unattended bag or suitcase on the carriageway, not carried (no person box on it), ≥ 2 s | 0 events on the samples |
| `illegal_turn` | **off** | — | Needs the list of permitted manoeuvres per approach (road signs), which we do not have. The previous rule flagged every ordinary turn (10–15 per video) |
| `illegal_u_turn` | **off** | — | Same reason |
| `solid_line_crossing` | **off** | — | Needs the solid markings as polylines. The previous rule fired on vehicles touching islands and dividers, which is a different event |

To switch a class on or off, edit `ENABLED_CLASSES` in `src/config.py`.
