# Homework 3 - Algorithmic Methods of Data Mining

### Author: Viktoriia Vlasenko  
**University Email:** [vlasenko.2088928@studenti.uniroma1.it](mailto:vlasenko.2088928@studenti.uniroma1.it)

---

## Overview
This repository contains the solution for **Homework 3** of the course **Algorithmic Methods of Data Mining**.

### Repository Contents
**Notebook**:
- **`main.ipynb`**: A Jupyter Notebook containing Python solutions of Homework3.

**Custom modules**:
- **`crawler.py`**: A module that is used for web scraping the Michelin Restaurants and crawling pages.
- **`parser.py`**: A module for parsing .html pages to extract information about the Michelin Restaurants and saving it to a single tsv file.
- **`engine.py`**: A module with all functions of serach engines.
- **`visualize.py`**: A module for custom form rendering.

Since the notebook is not interactive on github I will attach screenshoots from my local notebook.

Also I will attach a link to collab notebook:
[https://colab.research.google.com/drive/1iGdtRJno3IjTArm4yiT6GWidQYgfV761?usp=sharing](https://colab.research.google.com/drive/1iGdtRJno3IjTArm4yiT6GWidQYgfV761?usp=sharing)

**2. Create search engines**

Also I create a single form for inputting the query text. I used it in 2.1 and 2.2.
<p align=center>
<img src="images\form_query.png">
</p>

**3. New Score. Example of the custom form and the output**

I created a custom form using widgets that unfortunately not visible in the github notebook. It is possible to enter the query (accept chages or reset seletion), choose facilities/services, price range and cuisine type. I used this form in the points 3 and 4.
<p align=center>
<img src="images\metrics.png">
</p>
After clicking on Search button the the function render the resulting dataframe with the match.
<p align=center>
<img src="images\result.png">
</p>

**4. Create a Map**

The final map contains a heatmap defined by price ranges for each cities. In the legenda there are description of colors. Also there are present the restaurants' markers with info that were filtered from the restaurants' dataset by new score metrics.
<p align=center>
<img src="images\points.png">
</p>


**Preview of working visual elements in the notebook:**

https://github.com/user-attachments/assets/10c68069-365b-46fb-9577-2eeefbab1b39



