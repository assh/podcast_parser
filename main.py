# Import necessary libraries
import requests
import os
import json
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.mp3 import MP3
from PIL import Image
from io import BytesIO
import mutagen.id3
from bs4 import BeautifulSoup

# Function to download podcast episodes from an RSS feed
def download_podcast(rss_url, download_folder, history_file):
    # Fetch the RSS feed
    response = requests.get(rss_url)
    if response.status_code != 200:
        print("Failed to fetch RSS feed. Please check the URL.")
        return

    # Parse the RSS feed using BeautifulSoup
    soup = BeautifulSoup(response.content, 'lxml-xml')
    items = soup.find_all('item')

    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Load download history
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            downloaded_episodes = json.load(file)
    else:
        downloaded_episodes = []

    # Iterate over the episodes and download them
    for item in items:
        # Get episode title, GUID, and media URL
        title = item.find('title').text.replace('/', '-')  # Replace slashes to avoid issues with filenames
        guid = item.find('guid').text if item.find('guid') else title  # Use GUID if available, otherwise use title
        enclosure = item.find('enclosure')
        media_url = enclosure['url'] if enclosure else None

        if media_url and guid not in downloaded_episodes:
            print(f"Downloading: {title}")
            response = requests.get(media_url, stream=True)

            # Save the file to the download folder
            file_path = os.path.join(download_folder, f"{title}.mp3")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)

            # Add metadata to the downloaded file
            try:
                audio = EasyID3(file_path)
            except mutagen.id3.ID3NoHeaderError:
                audio = mutagen.File(file_path, easy=True)
                audio.add_tags()

            audio['title'] = title
            audio['artist'] = item.find('author').text if item.find('author') else 'Unknown Artist'
            audio['album'] = soup.find('title').text if soup.find('title') else 'Podcast'
            audio['date'] = item.find('pubDate').text if item.find('pubDate') else 'Unknown Date'
            audio['website'] = item.find('link').text if item.find('link') else ''
            audio.save()

            # Add album art to the downloaded file using episode-specific image
            image_tag = item.find('itunes:image')
            image_url = image_tag['href'] if image_tag else None

            if image_url:
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    img_byte_array = BytesIO(image_response.content).getvalue()

                    audio = MP3(file_path, ID3=mutagen.id3.ID3)
                    audio.tags.add(
                        mutagen.id3.APIC(
                            encoding=3,  # UTF-8
                            mime='image/jpeg',
                            type=3,  # Cover (front)
                            desc='Cover',
                            data=img_byte_array
                        )
                    )
                    audio.save()

            print(f"Downloaded and tagged: {title}\n")
            downloaded_episodes.append(guid)
        else:
            print(f"Skipping already downloaded episode: {title}\n")

    # Save updated download history
    with open(history_file, 'w') as file:
        json.dump(downloaded_episodes, file)

# Example usage
rss_feed_url = "https://feeds.simplecast.com/TBotaapn"  # Replace with the RSS feed URL of the podcast
download_folder = "podcasts"  # Replace with your desired download folder
history_file = "download_history.json"  # File to keep track of downloaded episodes

# Download podcast episodes
download_podcast(rss_feed_url, download_folder, history_file)
