from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd


# Parse tickers from the site
def parse(driver):
    companies_list = []
    old_companies_list = []

    while True:
        soup = BeautifulSoup(driver.page_source, 'lxml')

        # Find all the values in the rows
        all_values_div = soup.find_all('div', class_='jqx-grid-cell-left-align')


        all_values = []

        for i in all_values_div:
            all_values.append(i.text)

        # Select only ticker values
        companies_list.extend(all_values[1::3])

        # Check if the parser has reached the end
        if set(old_companies_list) == set(companies_list):

            return set(companies_list)

        old_companies_list.extend(companies_list)

        # Next page button
        driver.find_elements(By.CLASS_NAME, "jqx-icon-arrow-right")[1].click()

# Set a specific option, prepare for parsing
def set_options(driver, filter_index):
    url = 'https://www.macrotrends.net/stocks/stock-screener'
    driver.get(url)
    time.sleep(2)

    # Create link-list of 4 search fields
    Search_list = driver.find_elements(By.CLASS_NAME, "select2-search__field")

    # Open the second field
    Search_list[1].click()

    time.sleep(1)

    # Select USA country
    Country = driver.find_element(By.CLASS_NAME, "select2-results__option--highlighted")
    Country.click()

    time.sleep(1)

    # Open the third field
    Search_list[2].click()
    time.sleep(1)

    # Create a link-list of all sectors
    Fields = driver.find_elements(By.CLASS_NAME, "select2-results__option")

    # Select the appropriate fields until the all_list option
    if filter_index != -1:
        Fields[filter_index].click()

def main():
    #driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    driver = webdriver.Chrome()

    # Set a dict, Name : Fields index
    all_filters = {'Aerospace' : 0, 'Auto/Tires/Trucks' : 1, 'Basic Materials' : 2, 'Business Services' : 3, 'Computer and Technology' : 4, 'Construction' : 5, 'Consumer Discretionary' : 6, 'Consumer Staples' : 7, 'Finance' : 8, 'Industrial Products' : 9, 'Medical, Multi-Sector Conglomerates' : 10, 'Oils/Energy' : 11, 'Retail/Wholesale' : 12, 'Transportation' : 13, 'Unclassified' : 14, 'Utilities' : 15, 'Full_list' : -1}

    # Correct names to save a csv list
    all_filters_system = ['Aerospace', 'Auto_Tires_Trucks', 'Basic_Materials', 'Business_Services', 'Computer_and_Technology', 'Construction', 'Consumer_Discretionary', 'Consumer_Staples', 'Finance', 'Industrial_Products', 'Medical_Multi-Sector_Conglomerates', 'Oils_Energy', 'Retail_Wholesale', 'Transportation', 'Unclassified', 'Utilities', 'Full_list']

    for filter in all_filters:

        # Get Fields index
        filter_index = all_filters[filter]

        set_options(driver, filter_index)
        answ = parse(driver)

        # Refresh chrome for avoiding captcha
        driver.close()
        driver = webdriver.Chrome()

        # Create a pandas DataFrame
        data = pd.DataFrame(answ)
        data = data.set_axis([filter], axis=1)

        # Save the current data to a file with the correct name
        data.to_csv(rf'All_lists\{all_filters_system[filter_index]}_NOW.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()