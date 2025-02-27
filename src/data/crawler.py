from selenium import webdriver

# set the path to the web driver
driver_path = "/path/to/chromedriver"

# create a new Chrome browser instance
browser = webdriver.Chrome(executable_path=driver_path)

# navigate to a web page
browser.get("https://www.example.com")

# close the browser
browser.quit()
