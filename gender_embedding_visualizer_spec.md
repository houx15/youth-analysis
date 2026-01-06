# Gender Bias in Chinese Social Media: A Word Embedding Analysis
## Final Report Slide Outline

---

## I. Introduction

### Slide 1: Title Slide
- Title: Gender Bias in Chinese Social Media: A Word Embedding Analysis
- Subtitle: Evidence from 2024 Weibo Data
- Author & Date

### Slide 2: Research Questions
- How does occupational gender segregation vary across Chinese provinces?
- How are domestic work and labor differentially associated with gender?
- What regional patterns emerge in gender bias?

### Slide 3: Research Background
- Gender stereotypes in digital spaces
- Regional variation in gender attitudes
- Word embeddings as a tool for measuring implicit bias

---

## II. Data & Methods

### Slide 4: Data
- Source: 2024 Weibo posts
- Geographic coverage: Province-level analysis (based on IP)
- Word lists: Occupation terms, domestic work terms, labor terms
- Extraction based on vocabulary list

### Slide 5: Methodological Approach
- Word2Vec embeddings trained on Weibo corpus
- Two measurement methods:
  - **Similarity-based**: bias = female_similarity - male_similarity
  - **Projection-based**: bias = female_projection - male_projection
- Segregation index: Standard deviation of bias scores across provinces

---

## III. Results Part 1: Occupational Gender Segregation

### Slide 6: Overview - Similarity Method
- **Geographic distribution of occupational segregation**
  - Province-level map (darker = higher segregation)
  - Interpretation of regional patterns

### Slide 7: Provincial Ranking - Similarity Method
- **Provinces ranked by occupational gender segregation**
  - Horizontal bar plot showing all provinces
  - Highlighting top and bottom 5 provinces

### Slide 8: Top 5 Provinces Deep Dive - Similarity Method
- **Gender-typed occupations in highest segregation provinces**
  - For each of top 5 provinces:
    - Top 5 female-associated occupations
    - Top 5 male-associated occupations
  - Bar plots comparing bias scores

### Slide 9: Most Gender-Typed Occupations - Similarity Method
- **National-level occupation ranking**
  - Most female-associated occupations (by average similarity)
  - Most male-associated occupations (by average similarity)
  - Cross-provincial consistency

### Slide 10: Case Studies - Similarity Method
- **Regional variation in specific occupations**
  - Female-typed: 护士 (nurse), 幼师 (kindergarten teacher), 保姆 (nanny)
  - Male-typed: 程序员 (programmer), 工程师 (engineer), 保安 (security guard)
  - Mixed/emerging: 产品经理 (product manager), UP主 (content creator), 滴滴司机 (rideshare driver)
  - Province-by-province bias scores for each occupation

### Slide 11: Overview - Projection Method
- **Geographic distribution of occupational segregation (projection)**
  - Province-level map
  - Comparison with similarity method results

### Slide 12: Provincial Ranking - Projection Method
- **Provinces ranked by occupational gender segregation (projection)**
  - Horizontal bar plot
  - Method comparison

### Slide 13: Top 5 Provinces Deep Dive - Projection Method
- **Gender-typed occupations in highest segregation provinces (projection)**
  - Replication of Slide 8 analysis with projection method

### Slide 14: Most Gender-Typed Occupations - Projection Method
- **National-level occupation ranking (projection)**
  - Comparison with similarity method results

### Slide 15: Case Studies - Projection Method
- **Regional variation in specific occupations (projection)**
  - Same occupations as Slide 10, projection method
  - Method comparison insights

---

## IV. Results Part 2: Domestic Work/Labor & Gender

### Slide 16: Overview - Similarity Method
- **Geographic distribution of domestic/labor gender associations**
  - Province-level map
  - Bias = (domestic & female similarity) + (labor & male similarity)

### Slide 17: Provincial Ranking - Similarity Method
- **Provinces ranked by domestic/labor gender bias**
  - Horizontal bar plot showing all provinces

### Slide 18: Top 5 Provinces Analysis - Similarity Method
- **Gender associations in highest bias provinces**
  - For each of top 5 provinces:
    - Domestic & female association score
    - Labor & male association score
    - Top 5 domestic words (female-associated)
    - Top 5 labor words (male-associated)

### Slide 19: Word-Level Analysis - Similarity Method
- **Top domestic and labor terms**
  - Top 5 domestic words: province-by-province bias scores
  - Top 5 labor words: province-by-province bias scores
  - Regional variation patterns

### Slide 20: Overview - Projection Method
- **Geographic distribution of domestic/labor gender associations (projection)**
  - Province-level map
  - Bias = domestic female projection - labor male projection
  - Comparison with similarity method

### Slide 21: Provincial Ranking - Projection Method
- **Provinces ranked by domestic/labor gender bias (projection)**
  - Horizontal bar plot
  - Method comparison

### Slide 22: Top 5 Provinces Analysis - Projection Method
- **Gender associations in highest bias provinces (projection)**
  - Replication of Slide 18 analysis with projection method

### Slide 23: Word-Level Analysis - Projection Method
- **Top domestic and labor terms (projection)**
  - Replication of Slide 19 analysis with projection method
  - Cross-method insights

---

## V. Discussion & Conclusion

### Slide 24: Key Findings
- Regional variation in occupational gender segregation
- Persistent associations between domestic work/femininity and labor/masculinity
- Consistency across measurement methods
- Specific provinces with highest/lowest bias

### Slide 25: Theoretical Implications
- Gender stereotypes in digital discourse
- Regional heterogeneity in gender attitudes
- Methodological contributions (similarity vs. projection)

### Slide 26: Limitations & Future Directions
- Social media data limitations
- Causality and interpretation
- Future research directions

### Slide 27: Conclusion
- Summary of main contributions
- Practical implications

---

## Notes for Presentation:
- Each results slide should include the corresponding visualization from the analysis
- Use consistent color schemes across similarity and projection methods
- Highlight key differences between methods where relevant
- Consider backup slides with additional case studies or sensitivity analyses
