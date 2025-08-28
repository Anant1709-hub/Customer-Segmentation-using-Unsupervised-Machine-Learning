# 🛍️ Customer Segmentation using Unsupervised Machine Learning  

Customer Segmentation involves grouping customers based on shared characteristics, behaviors, and preferences. By segmenting customers, businesses can **tailor marketing strategies**, **personalize customer experience**, and **enhance overall business value**.  

This project uses **Unsupervised Machine Learning (K-Means Clustering + t-SNE visualization)** in Python to segment customers from a retail dataset.  

---

## 📌 Project Workflow  

### **1. Import Libraries**  
We use Python libraries such as **Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn** for preprocessing, visualization, and clustering.  

### **2. Load Dataset**  
- Dataset contains customer details (income, marital status, purchases, etc.).  
- Shape: `(2240, 29)` before cleaning.  

### **3. Data Preprocessing**  
- Handle missing values (drop small % of nulls).  
- Convert date features (`Dt_Customer`) into **day, month, year**.  
- Drop irrelevant columns (`Z_CostContact`, `Z_Revenue`, `Dt_Customer`).  
- Encode categorical variables using **Label Encoding**.  
- Standardize features with **StandardScaler**.  

### **4. Data Visualization**  
- **Count plots** for categorical variables.  
- **Heatmap** for correlation analysis.  
- **t-SNE** for dimensionality reduction and visual exploration.  

### **5. Clustering (Customer Segmentation)**  
- Used **K-Means Clustering**.  
- Optimal number of clusters found using **Elbow Method** → **k = 6**.  
- Visualized clusters using **t-SNE scatter plot**.  

---

## 📊 Results  
- Customers are successfully divided into **6 meaningful segments**.  
- Segmentation helps identify groups of customers with **similar purchasing behavior**.  
- Businesses can use these clusters for:  
  - Targeted marketing campaigns  
  - Personalized offers and ads  
  - Better resource allocation  

---

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  

---

## 🚀 How to Run  

1. Clone this repository:  
   ```
   git clone https://github.com/your-username/customer-segmentation.git
   cd customer-segmentation
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Place the dataset (new.csv) in the project folder.

4. Run the script / Jupyter Notebook.

✨ Conclusion

This project demonstrates how unsupervised learning can be applied for customer segmentation, providing valuable insights for businesses to personalize strategies and maximize growth.
