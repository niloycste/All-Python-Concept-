from selenium.webdriver.common.by import By
from selenium import webdriver
from tqdm import tqdm
import pandas as pd
import time

if __name__ == "__main__":
    webdriver_path = "C:\Program Files (x86)\chromedriver.exe"
    driver = webdriver.Chrome(webdriver_path)

    df = pd.read_csv("book_urls.csv")
    book_urls = df.url.to_list()

    book_data = []
    for book_url in tqdm(book_urls):
            
        try:
            driver.get(book_url)
            time.sleep(3)

            title_element = driver.find_element(By.CLASS_NAME, "BookPageTitleSection")
            title = title_element.find_element(By.TAG_NAME, "h1").text

            description = driver.find_element(By.CLASS_NAME, "BookPageMetadataSection__description").text
            description = description.replace("\n", "").replace("Show more", "")

            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if button.text == "...more":
                    button.click()
                    time.sleep(3)
                    break
            
            genres = driver.find_element(By.CLASS_NAME, "BookPageMetadataSection__genres").text
            genres = genres.split("\n")[1:-1]

            book_data.append({
                "title": title,
                "url": book_url,
                "description": description,
                "genres": genres
            })
            time.sleep(3)

            df = pd.DataFrame(data=book_data, columns=book_data[0].keys())
            df.to_csv("book_details.csv", index=False)
        
        except:
            time.sleep(3)

