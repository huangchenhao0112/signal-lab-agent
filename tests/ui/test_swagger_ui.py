from __future__ import annotations

import pytest

from selenium import webdriver

from selenium.webdriver.common.by import (
    By,
)

from selenium.webdriver.support.ui import (
    WebDriverWait,
)

from selenium.webdriver.support import (
    expected_conditions as EC,
)


pytestmark = pytest.mark.ui


SWAGGER_URL = (
    "http://localhost/api/docs"
)


@pytest.fixture
def driver():
    options = webdriver.ChromeOptions()

    options.add_argument(
        "--window-size=1400,1000"
    )

    browser = webdriver.Chrome(
        options=options
    )

    yield browser

    browser.quit()


def test_swagger_page_loads(
    driver,
):
    driver.get(
        SWAGGER_URL
    )

    wait = WebDriverWait(
        driver,
        10,
    )

    wait.until(
        EC.presence_of_element_located(
            (
                By.CLASS_NAME,
                "swagger-ui",
            )
        )
    )

    body = driver.find_element(
        By.TAG_NAME,
        "body",
    ).text

    assert "/agent/health" in body
    assert "/agent/chat" in body


def test_swagger_contains_tools_endpoint(
    driver,
):
    driver.get(
        SWAGGER_URL
    )

    wait = WebDriverWait(
        driver,
        10,
    )

    wait.until(
        EC.presence_of_element_located(
            (
                By.CLASS_NAME,
                "swagger-ui",
            )
        )
    )

    body = driver.find_element(
        By.TAG_NAME,
        "body",
    ).text

    assert "/agent/tools" in body