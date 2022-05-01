db.processed_crypto_data.aggregate(
    {
        $project: {
            _id: 1, Ticker_timestamp: 1, Ticker_timestamp_iso: 1
        }
    })

// find latest crypto data in collection
db.processed_crypto_data.find().sort({"Ticker_timestamp": -1}).limit(1)

// find latest reddit post in collection
db.reddit_crypto_data.find().sort({"created_unix": -1}).limit(1)