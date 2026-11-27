# Contributions

*Fill this in as you go, not on the last day. It is checked against the commits and against the peer assessment. A first name and a GitHub username are enough: this repo is public.*

## Members

| First name | GitHub username | Main areas |
|---|---|---|
| Lucía | luciamunozchicote | Repo and deployment on Render; component 1 (Gemini vision) and labelling of 1,000 DeepSolarEye photos; component 2 (regression); € decision threshold; monitoring and drift check; evaluation code; the 45 test cases and their photos; End to end section |
| Diana | Dianadelorenzo | Architecture, Operations and How to run it sections; drift alert screenshot |
| Alejandra | Alejandraahe | Evaluation of components, and Cost and latency sections |
| Celia | celiasanmartin | Data and Failures sections (sources, licences, personal-data review, failure analysis) |
| María | mariamartinezgonzalez | Business case and Decision sections; editing of the report |
| Louis | louisdeperier123 | Presentation slides and live demo |

## Log

*One line per piece of work: the date, who did it (more than one name is fine), and what. Code done in pair programming counts for both, even if only one of you committed it.*

| Date | Who | What |
|---|---|---|
| 18 Nov | All | Chose the problem (solar panel maintenance) and the two components; approved the design decisions |
| 18 Nov | Lucía | Set up the repo and the Render app; component 1: Gemini describes the panel (vision.py) |
| 18–20 Nov | Lucía | Labelled 1,000 DeepSolarEye photos with Gemini (tools/label_deepsolareye.py) |
| 19–20 Nov | Lucía | Component 2: trained and compared the loss regressions (component2.py, tools/train_component2.py) |
| 20 Nov | Lucía | € decision threshold (economics.py), app actions, drift sets, monitoring signals and evaluation code |
| 23 Nov | Lucía | Wrote the 45 test cases, chose and licensed the photos (eval/cases.csv, SOURCES.md) |
| 23 Nov | Lucía, Louis | Took the team decisions: baseline, days between cleanings, review and drift thresholds, no RL bonus |
| 24 Nov | Lucía | Ran the full evaluation, drift demo and checks; timing per component; privacy fixes in the logs |
| 24 Nov | María | Business case and Decision sections; added herself to contributions |
| 24 Nov | Celia | Data and Failures sections |
| 24 Nov | Alejandra | Evaluation: components, and Cost and latency |
| 24 Nov | Diana | Architecture, Operations and How to run it; drift alert screenshot |
| 25 Nov | Louis, Lucía | End to end section (draft written with AI help, see AI use); fact-checked and corrected report numbers |
| TODO | Louis | Presentation slides and live demo |
| TODO | María, Louis | AI use section; final edit of the report to ~3,000 words |

AI assistance (Claude) was used for the code and for checking the report's numbers; see "AI use" in REPORT.md.
