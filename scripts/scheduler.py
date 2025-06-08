import schedule
import time
from src.scraper import scrape_all_banks


def configure_schedule(option="daily"):
    """
    Configure the scraping schedule.

    Args:
        option (str): Schedule option ['daily', 'hourly', '6hr', 'weekly']
    """
    if option == "daily":
        schedule.every().day.at("01:00").do(scrape_all_banks)
    elif option == "hourly":
        schedule.every().hour.do(scrape_all_banks)
    elif option == "6hr":
        schedule.every(6).hours.do(scrape_all_banks)
    elif option == "weekly":
        schedule.every().monday.at("08:00").do(scrape_all_banks)
    else:
        raise ValueError("❌ Invalid schedule option. Choose from 'daily', 'hourly', '6hr', 'weekly'.")


if __name__ == "__main__":
    # You can change the schedule option here
    configure_schedule(option="daily")
    print("📅 Scheduler is running... Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)
