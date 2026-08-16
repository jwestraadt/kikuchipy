# Patent search record -- 2026-08-16

Purpose: check whether the "provisional patent application" mentioned in the
EMSphInx README ("The central indexing algorithm is covered by a provisional
patent application", `EMSphInx/ReadMe.md`, Financial Support section; EMSphInx
first released 2019, © De Graef Group, Carnegie Mellon University, author
W. C. Lenthe) matured into a published application or granted patent.

Method: Google Patents' JSON search endpoint (`https://patents.google.com/xhr/query?url=<encoded query>`),
plus general web searches. All queries run on 2026-08-16 from Columbus, OH (US).
This is a bounded search, not a freedom-to-operate opinion.

## Queries and results

| # | Query (decoded) | Total | Relevant hits |
|---|---|---|---|
| A | `q="spherical harmonic" "backscatter" indexing inventor:Lenthe` | 2 | WO2025184557A1; US20260002896A1 (both below) |
| B | `q="spherical" "diffraction" "cross-correlation" assignee:"Carnegie Mellon"` | 1 | US20250060244A1 -- "System, Method, and Computer Program Product for Optical Vibration Sensing", CMU, inventors Sheinin/O'Toole/Narasimhan/Chan, priority 2021-12-17 (unrelated) |
| C | `q="electron backscatter" "spherical harmonic"` (num=100) | 42 | Only two with inventor Lenthe/Singh/De Graef or assignee CMU: WO2026010991A1 and WO2025184557A1 (below). Remainder are metallurgy/alloy patents that mention EBSD texture (e.g. JP5158909B2, TWI732529B, US9803269B2). |
| D | Web search: `patent Lenthe De Graef Carnegie Mellon spherical harmonic indexing electron backscatter diffraction cross-correlation` | -- | Only the 2019 Ultramicroscopy and J. Appl. Cryst. papers and derivative literature; no patent documents. |
| E | Web search restricted to patents.google.com / justia / uspto / freepatentsonline: `"spherical harmonic" EBSD indexing patent Carnegie Mellon Lenthe "De Graef"` | -- | No CMU/De Graef/Lenthe patent; unrelated hits US9070203B2, US7442930B2, US20130208951A1. |

## Filings surfaced for inventor William Carl Lenthe (all post-CMU, other assignees)

| Publication | Title | Assignee | Priority | Published |
|---|---|---|---|---|
| WO2025184557A1 | Devices and systems for experimental forward model indexing of electron backscatter diffraction patterns | Gatan Inc. | 2024-03-01 | 2025-09-04 |
| US20260002896A1 / WO2026010991A1 | Systems and methods for frame control in texture analysis | Edax LLC / Edax Inc. | 2024-07-01 | 2026-01-01 |

Neither claims the 2019 CMU spherical-harmonic cross-correlation algorithm
(different assignee, 2024 priority; WO2025184557A1 concerns experimental
forward-model indexing). They are noted because they show the inventor's later
commercial work in EBSD indexing; whether their claims read on any part of the
port is a question for counsel, not for this project.

## Conclusion

No granted or published patent on the EMSphInx spherical indexing algorithm
was found for inventors Lenthe, Singh or De Graef or assignee Carnegie Mellon
University. A US provisional filed in 2019 that was converted to a
non-provisional would ordinarily have published within 18 months of priority
(by ~2021); none appears. Residual uncertainty remains (unpublished/abandoned
prosecution, non-US filings not surfaced by these queries), which is why the
constitution flags the statement to the pyxem maintainers before any upstream
merge and keeps the CMU CTTEC contact (`innovation@cmu.edu`) in the notices.
