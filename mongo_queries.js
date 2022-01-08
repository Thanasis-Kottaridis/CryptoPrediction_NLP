db.processed_crypto_data.aggregate(
    {
        $project: {
            _id: 1, Ticker_timestamp: 1, Ticker_timestamp_iso: 1
        }
    })


db.processed_crypto_data.find().sort({"Ticker_timestamp": -1}).limit(1)
