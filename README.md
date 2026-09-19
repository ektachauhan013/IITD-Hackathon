# Corporate Heist

## Candidate Ranking & Selection System

### Problem

Hiring decisions involve evaluating candidates across multiple dimensions such as technical ability, experience, career history, role suitability, and professional background.

The challenge is to build a data-driven candidate ranking system that can identify strong candidates while accounting for the fact that hiring patterns may change over time.

This project focuses on building a ranking pipeline that combines **historical hiring information with current-cycle signals and candidate-level consistency checks** to produce a reliable candidate shortlist.

---

## Approach

The project follows a structured candidate-ranking pipeline:

1. **Data Preparation**

   * Clean and preprocess candidate information.
   * Handle missing values and inconsistent representations.
   * Normalize relevant categorical fields.
   * Prepare numerical and categorical features for modeling.

2. **Historical Pattern Analysis**

   * Study previous hiring outcomes to understand patterns associated with successful candidates.
   * Use historical information as supporting evidence rather than treating past hiring behavior as an absolute rule.

3. **Feature Engineering**

   * Construct meaningful candidate-level features from available attributes.
   * Capture experience, career progression, role alignment, professional history, and other relevant characteristics.

4. **Current-Cycle Signals**

   * Incorporate signals that are specific to the current hiring cycle.
   * Give appropriate consideration to information that was not available in earlier hiring cycles.

5. **Profile Consistency**

   * Check for potentially inconsistent candidate information.
   * Identify unusual combinations of age, education, experience, and career progression.
   * Treat these as signals for further evaluation rather than automatically assuming fraud.

6. **Duplicate Handling**

   * Detect candidates who may appear under multiple candidate IDs.
   * Prevent the same individual from being selected multiple times.

7. **Candidate Ranking**

   * Generate a candidate score/ranking using the combined signals.
   * Apply relevant constraints and selection logic.
   * Produce the final ranked candidate set.

---

## Key Design Principles

### 1. Historical Data Is Evidence, Not a Rule

Previous hiring outcomes can reveal useful patterns, but blindly reproducing historical decisions can carry forward outdated preferences.

The system therefore uses historical information as one component of the ranking process.

### 2. Current Signals Matter

The current hiring cycle can introduce new information and different priorities.

The ranking process is designed to incorporate these signals rather than relying entirely on historical patterns.

### 3. Technical and Role Relevance

Candidate suitability should be considered in the context of the role being applied for.

Relevant technical background, experience, and role alignment are therefore important components of the evaluation.

### 4. Profile Consistency

Candidate information should form a reasonably consistent professional profile.

Potential inconsistencies are treated as risk signals and evaluated alongside the rest of the candidate information.

### 5. Duplicate-Aware Selection

A candidate appearing under multiple IDs should not result in multiple selections of the same individual.

Identity-level duplicate handling is therefore included before the final shortlist is formed.

### 6. Avoiding Outdated Biases

The system avoids blindly relying on factors such as:

* College prestige
* City or location prestige
* Employer brand
* Referrals
* Conventional educational backgrounds

These factors should not automatically outweigh technical ability, role relevance, and current-cycle signals.

---

## Methodology

The overall methodology can be summarized as:

```text
Candidate Data
      ↓
Data Cleaning & Normalization
      ↓
Feature Engineering
      ↓
Historical Signal Extraction
      ↓
Current-Cycle Signal Integration
      ↓
Profile Consistency Checks
      ↓
Duplicate Handling
      ↓
Candidate Scoring & Ranking
      ↓
Final Shortlist
```

The detailed analysis, feature reasoning, and modeling methodology are documented separately in `documentation.pdf`.

---

## Key Takeaways

* Candidate ranking should combine **multiple sources of evidence** rather than depend on a single feature.
* Historical hiring patterns are useful, but they should not automatically define current hiring decisions.
* **Current-cycle information** can be more relevant than historical patterns when hiring priorities change.
* Candidate quality should be evaluated in the context of the **role and technical requirements**.
* Data-quality and profile-consistency checks can help identify unreliable candidate records.
* Duplicate detection is important when candidate identity can be represented by multiple IDs.
* A robust ranking system should balance **predictive signals, current requirements, and practical hiring constraints**.
* The final system is designed to produce a ranking that is both **data-driven and adaptable to changing hiring requirements**.
