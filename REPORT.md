# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render):
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

## Business case

1. Who uses it and what they decide

It's used by the maintenance manager of an operations and maintenance (O&M) company that manages several medium or large solar plants in southern Spain.
From a photo of a panel, they decide whether to clean it, repair it, or do nothing.
The app decides per photo. It doesn't prioritize between panels or between plants.

2. What happens today (baseline)

Panels are cleaned on a fixed schedule, whether they need it or not.
Or the company reacts when production drops. Monitoring warns that output is falling, but doesn't say why or which panel, so someone has to be sent to check.
In the evaluation, the baseline is "wait for the scheduled cleaning". We also compare against "always clean".

3. Why the app is useful

From one photo, it tells apart dirt, shade and damage, and each one is fixed differently.
Shade can't be fixed by cleaning, and a crack needs a technician for safety and warranty reasons.
In southern Spain there is dust, calima (Saharan dust haze) and rain-free summers, so dirt is a real problem.

4. How we measure success

Percentage of correct decisions across 45 cases versus the baseline: the app is right 67% of the time, "wait" 42% and "always clean" 31%.
Undetected damage. The target is 0, and the app caught 5 out of 5.
Unnecessary cleanings.

5. How much it's worth (in €)

Cleaning is worth it when the energy lost until the next cleaning is worth more than the cost of cleaning.
Team assumptions: cleaning costs €0.09 per m², there are 30 days until the next cleaning, the panels are 21% efficient, and energy is worth €0.0325 per kWh.
Result: cleaning pays off if the loss is above 9.3%.
Sensitivity: between 8.3% and 10.2% depending on the energy price; 18.7% with 15 days until cleaning and 4.7% with 60.
Example: a 50 MW plant with a 5% loss loses about €380 per day at that price.
Cost of using the app: about €0.00136 per photo. With 500 photos a day that's €0.68 a day, about €20 a month.
Limit: the app decides per photo and doesn't prioritize between plants.

## Architecture

*The two components and how one feeds the other: a diagram plus a short walk-through of one request from input to output. Which models or APIs, and why those. What you reused from assignments 1–3, and whose it was.*

## Data

**Component 1 (Gemini vision).** It is not trained: it receives a photo and returns a structured description (soiled or clean, number of spots, share of the panel covered, how the dirt is spread, and whether the image shows a solar panel).

**Component 2 (loss model).** It is trained on 1,000 photos from DeepSolarEye (Mehta et al., WACV 2018, licence CC BY-NC-SA, https://deep-solar-eye.github.io). Each photo comes with its measured power loss and the irradiance and time of the shot. Gemini described all 1,000 photos, and those descriptions plus the context are the model's inputs. We split by day: 768 photos for training and 232 for testing (27–30 June 2017), so the test photos come from days the model never saw. The photos are not in the repository, only the labels (`data/deepsolareye_gemini.csv`); they must be downloaded from the dataset's site. The "non-commercial" licence is fine for this coursework, but a real product would need another dataset or permission.

**Evaluation and monitoring sets.**
- 45 end-to-end cases (5.9 MB): 23 DeepSolarEye photos and 22 photos from Flickr and Wikimedia Commons under CC BY 2.0, CC BY-SA 2.0 or CC BY-SA 4.0. Author, licence and link for each are in `eval/inputs/cases/SOURCES.md`.
- 40 reference photos (812 KB) for the drift check, all from DeepSolarEye.
- 15 shifted photos (228 KB) for the drift demo, DeepSolarEye photos altered on purpose (for example, haze).

**Personal data.** The images show only panels, roofs and landscapes: no faces, names, number plates or addresses. The logger stores a text summary of the input, the outputs, versions, time, cost and the user's thumbs up/down, not the image itself. There is no personal data in the inputs or in the logs.

## Evaluation

### Components

*Each component on its own: the metric, the result, the test data and its size.*

### End to end

*How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the baseline. Where the output is free text: the judge's rubric and your agreement with it on your own labels.*

### Cost and latency

*Per request: tokens, euros and seconds, for each component and in total. What it would cost per month at the volume in your business case.*

## Operations

*What you log. The quality signal and the drift check you monitor, the reference set, and the alert threshold with its justification. The drift demo: what you shifted, and the result with the alert firing (screenshot or numbers). The rate of false alarms on normal inputs.*

## Failures

The product failed on 15 of the 45 cases (67% correct). All 15 start at component 1 (Gemini), and for most of them component 2 then amplifies the error.

| Cause | Cases | Count |
|---|---|---|
| Nearly clean panel seen as soiled | c01–c07, c20, c21, c23, c37 | 11 |
| Shadow mistaken for dirt (expected: check shade) | c41, c42 | 2 |
| Polycrystalline texture mistaken for stains | c38 | 1 |
| Solar water heater taken for a panel | c29 | 1 |

The largest group is 11 of 15 failures (73%). In those cases the true loss is below the 9.3% at which cleaning pays off (0.3% to 9.2% on the DeepSolarEye photos), so the correct action was to wait. Gemini reports only 1–5% of the panel covered, yet it still labels the panel "soiled", and the loss model turns that label into an estimated loss of 17–31%. The result is an unnecessary cleaning order. This costs money but does not put the panels at risk, and one of the 11 (c23) was sent to a person because Gemini's three votes disagreed.

**What we would fix first:** this group. We would add examples of nearly clean panels to Gemini's prompt, add a "lightly soiled" level, and send a case to a person when Gemini says "soiled" but reports 5% coverage or less. Fixing only these 11 would lift accuracy from 67% to up to 91% (41 of 45). The shadow and out-of-scope failures are only 4 cases, so we would handle them later, with a "shaded" option and a stricter check that the image is a PV panel.

## Decision
1. Which cases go to a person

Visible damage, always.
Cases where Gemini's three answers have no majority.
An "unknown" answer.
Gemini not available.

2. How many cases per day

Of the 45 test cases, 7 go to a person (16%): the 5 damage cases, plus c23 and c34. Without the 5 damage cases, which were chosen on purpose, it's 2 out of 40 (5%).
We assume 500 photos a day, a sample of areas across the plants. This is an assumption and can be changed.
5% is about 25 cases a day, not counting damage.

3. Evidence in favor

It's right in 30 of 45 cases (67%), versus 42% and 31%.
Damage 5/5, and panels that needed cleaning 14/14.
It costs about €0.00136 per photo and takes about 13.5 seconds.

4. Evidence against and limits

11 of the 15 mistakes are nearly clean panels that the app tells you to clean. That costs money but doesn't put the panels at risk.
Only 1 of those 15 mistakes reached a person, because Gemini is confidently wrong.
In the 33 dirt cases, the advantage over "wait" is small (64% vs 58%).
In euros per m², on the test days the model doesn't beat "always clean" (0.0282 vs 0.0259). In cross-validation it does (0.0182 vs 0.0208).
The dataset photos are small (192×192 pixels) and have a non-commercial license. A real plant and drone photos would look different.

5. Verdict: go with conditions

Extra review: if Gemini says "dirty" and 5% or less of the surface is covered, it goes to a person.
In the 1,000 labeled photos, that would be 34%. In 42% of those cases, waiting was the correct choice.
With 500 photos a day, that could be up to about 170 cases. This is a rough maximum, because those photos aren't real traffic.
Improve Gemini: add examples of nearly clean panels to the prompt, and add a "slightly dirty" level.
Pilot at one plant: a person decides in parallel, and the results are compared.
Move to a full "go": if in the pilot there are fewer unnecessary cleanings than with "wait" and no damage slips through.

## Bonus: reinforcement learning

*Delete this section if you did not do it. The loop (feedback → reward → update → behaviour), the reward and what it could reward by accident, the exploration strategy and its cost, the evidence from real logs (and which part was simulated), the risks.*

## How to run it

*Anything beyond SETUP.md that the professor needs to run it: which keys, any download step, how long the first run takes.*

## AI use

*What you used and how, or "none".*
