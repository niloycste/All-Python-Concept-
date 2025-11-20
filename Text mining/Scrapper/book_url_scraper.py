from selenium.webdriver.common.by import By
from selenium import webdriver
from tqdm import tqdm
import pandas as pd
import time


if __name__ == "__main__":
    webdriver_path = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(webdriver_path)
    base_url = "https://www.goodreads.com/list/show/264.Books_That_Everyone_Should_Read_At_Least_Once"
    book_urls = []
    
    for idx in tqdm(range(100)):

        page_no = idx + 1
        page_url = f"{base_url}?page={page_no}"
        driver.get(page_url)
        rows = driver.find_elements(By.TAG_NAME, 'tr')

        for row in rows:
            url_tag = row.find_element(By.CLASS_NAME, 'bookTitle')
            title = url_tag.text 
            book_url = url_tag.get_attribute('href')
            book_urls.append({
                "title": title,
                "url": book_url
            })
        
        time.sleep(1)
    
    df = pd.DataFrame(data=book_urls, columns=book_urls[0].keys())
    df.to_csv("book_urls.csv", index=False)


    
