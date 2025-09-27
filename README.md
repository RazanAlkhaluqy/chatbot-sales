# chatbot-sales

# 📦 Amazon Product Data Analysis & Chatbot

![Project Badge]![Python](https://img.shields.io/badge/Python-3.11-green) ![License](https://img.shields.io/badge/License-MIT-yellow)


Analyze Amazon product data using Python and Excel, perform sentiment analysis, and interact with insights through a simple AI-powered chatbot.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Dictionary](#data-dictionary)
- [Technology Stack](#technology-stack)
- [Installation & Setup](#installation--setup)
- [Analysis & Insights](#analysis--insights)
- [Chatbot Integration](#chatbot-integration)
- [Screenshots / Demo](#screenshots--demo)
- [License](#license)
- [Contributors](#contributors)

---

## Project Overview
This project focuses on exploring, analyzing, and extracting insights from the **Amazon Product Dataset** using Excel and Python. The project demonstrates how to:  
- Perform data cleaning and preprocessing using **Pandas**  
- Conduct exploratory data analysis (EDA) to answer business questions  
- Apply **AI models from Hugging Face** for sentiment analysis of reviews  
- Build a **basic chatbot** to answer queries about products and insights  

The ultimate goal is to provide actionable insights on product performance and customer sentiment while demonstrating AI-assisted analytics.

---

## Dataset
The dataset contains product and review information from Amazon. It includes **16 columns** with details like pricing, ratings, reviews, and user information.  

**Data Source:** Local Excel file (provided for Week 3 tasks)

### Data Dictionary
| Column | Description |
|--------|-------------|
| `product_id` | Unique identifier for each product |
| `product_name` | Full product name |
| `category` | Product category (may include multiple levels) |
| `discounted_price` | Price after discount |
| `actual_price` | Original price before discount |
| `discount_percentage` | Discount as a percentage |
| `rating` | Average product rating |
| `rating_count` | Total number of ratings |
| `about_product` | Short product description |
| `user_id` | Unique identifier of reviewer |
| `user_name` | Name of reviewer |
| `review_id` | Unique ID of the review |
| `review_title` | Title or summary of the review |
| `review_content` | Full review text |
| `img_link` | Link to product image |
| `product_link` | Link to product page |

---

## Technology Stack
- Python 3.11  
- Pandas, NumPy, Matplotlib, Seaborn  
- Jupyter Notebook  
- Streamlit (for dashboard)  
- Hugging Face Transformers (for sentiment analysis)  
- Git & GitHub (version control)

---

## Installation & Setup
1. **Clone the repository**
```bash
git clone https://github.com/yourusername/amazon-product-analysis.git
cd amazon-product-analysis

## Analysis & Insights

**Exploratory Analysis:**  
Examined product categories, top-rated products, price distributions, and rating trends.

**Business Questions:**  
Answered at least five business questions, such as top-selling categories and highest-rated products.

**Custom Analysis:**  
Explored additional questions, including sentiment trends across categories and the relationship between discount percentage and rating.

**Visualizations:**  
- Price vs Rating Scatter Plot  
- Rating Count Distribution by Category  
- Sentiment Analysis Heatmap

---

## Chatbot Integration

- A simple AI-powered chatbot using a local LLM  
- Allows users to query insights such as:  
  - Top products by category  
  - Average ratings  
  - Sentiment trends of reviews  

**Future Enhancements:** Could incorporate multi-turn conversation and advanced NLP for deeper insights.

---

## Screenshots / Demo

*Add screenshots or GIF here showcasing the dashboard, visualizations, and chatbot interaction.*

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contributors

- **Razan Zaki** – Data analysis, AI integration, Python & Excel handling  
- Open to contributions via Pull Requests


