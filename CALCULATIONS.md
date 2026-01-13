# CSVy - Statistical Calculations Reference

## Mathematical Formulas & Calculations

### 1. Normalization Methods

#### Min-Max Normalization (0-1 Scaling)
```
x_normalized = (x - min(X)) / (max(X) - min(X))
```

**Example:**
```
Input:  [10, 20, 30, 40, 50]

Normalized Output: [0.00, 0.25, 0.50, 0.75, 1.00]
```

#### Z-Score Standardization
```
z = (x - μ) / σ

where:
  μ = mean of dataset
  σ = standard deviation
```

**Example:**
```
Input:  [10, 20, 30, 40, 50]

Standardized Output: [-1.41, -0.71, 0.00, 0.71, 1.41]
```

---

### 2. Descriptive Statistics

#### Mean (Average)
```
μ = (Σ xi) / n = (x₁ + x₂ + ... + xₙ) / n
```

#### Median (Middle Value)
```
For odd n: median = x[(n+1)/2]
For even n: median = (x[n/2] + x[n/2+1]) / 2
```

#### Variance
```
σ² = Σ(xi - μ)² / n
```

#### Standard Deviation
```
σ = √(σ²) = √(Σ(xi - μ)² / n)
```

---

### 3. Quartiles & Percentiles

#### Quartile Calculation
```
Q1 = 25th percentile (1st quartile)
Q2 = 50th percentile (median)
Q3 = 75th percentile (3rd quartile)
IQR = Q3 - Q1 (Interquartile Range)
```

---

### 4. Outlier Detection

#### IQR Method
```
Lower Fence = Q1 - 1.5 × IQR
Upper Fence = Q3 + 1.5 × IQR

Outlier if: x < Lower Fence OR x > Upper Fence
```

#### Z-Score Method
```
z = (x - μ) / σ

Outlier if: |z| > 3 (typically)
```

---

### 5. Data Quality Metrics

#### Missing Value Rate
```
Missing Rate = (Missing Count / Total Count) × 100%
```

#### Cardinality
```
Cardinality = (Unique Values / Total Values) × 100%
```

#### Duplicate Detection
```
Duplicate Rate = (Duplicate Rows / Total Rows) × 100%
```

---

### 6. Correlation Analysis

#### Pearson Correlation Coefficient
```
r = Σ[(xi - μx)(yi - μy)] / √[Σ(xi - μx)² × Σ(yi - μy)²]

Range: -1 ≤ r ≤ 1
  r =  1: Perfect positive correlation
  r =  0: No correlation
  r = -1: Perfect negative correlation
```

---

### 7. Encoding Calculations

#### One-Hot Encoding
```
Original: Category column with n unique values
Result:   n binary columns (0 or 1)
Space complexity: O(n × m) where m = number of rows
```

#### Label Encoding
```
Original: n unique categories
Result:   Integers from 0 to (n-1)
Mapping: {Category1: 0, Category2: 1, ..., Categoryn: n-1}
```

---

### 8. Distance & Similarity Metrics

#### Euclidean Distance
```
d(p, q) = √[Σ(pi - qi)²]
```

#### Manhattan Distance
```
d(p, q) = Σ|pi - qi|
```

---

### Reference Table

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Mean** | `Σx / n` | Central tendency |
| **Median** | Middle value | Robust central tendency |
| **Std Dev** | `√(Σ(x-μ)²/n)` | Data spread |
| **Min-Max Norm** | `(x-min)/(max-min)` | Scale to [0,1] |
| **Z-Score** | `(x-μ)/σ` | Standardize data |
| **IQR** | `Q3 - Q1` | Outlier detection |
| **Pearson r** | `cov(X,Y)/(σx×σy)` | Linear correlation |
| **Missing Rate** | `(missing/total)×100%` | Data quality |

---

**Note**: All calculations in CSVy follow standard statistical formulas and best practices for data preprocessing.
