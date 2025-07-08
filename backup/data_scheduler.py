        # Run initial data fetch
        logger.info("Starting initial data fetch...")
        
        # Fetch historical data
        fetcher.fetch_daily_data(force_full_history=True)
        fetcher.fetch_minute_data(force_full_history=True)
        
        logger.info("Initial data fetch complete.")
        
        # Set up schedule and run continuously
        schedule = fetcher.schedule_daily_fetches()
        
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        while True:
            schedule.run_pending()
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    except Exception as e:
        logger.error(f"Error in scheduler: {e}", exc_info=True)

if __name__ == "__main__":
    main()