import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to retrieve the document")

    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.find_all("tr")  # Extract table rows

    grid = {}

    for row in rows[1:]:  # Skip the header row
        cols = row.find_all("td")
        if len(cols) == 3:
            x = cols[0].text.strip()
            char = cols[1].text.strip()
            y = cols[2].text.strip()

            if x.isdigit() and y.isdigit():
                grid[(int(x), int(y))] = char

    if not grid:
        raise ValueError("No valid grid data found. Ensure the document format is correct.")

    max_x = max(coord[0] for coord in grid.keys())
    max_y = max(coord[1] for coord in grid.keys())

    output_grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    for (x, y), char in grid.items():
        output_grid[y][x] = char  

    for row in output_grid:
        print("".join(row))

decode_secret_message("https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub")