exchange_calendars = {
    "US_AMERICA": {
        "United States": "XNYS"  # New York Stock Exchange (NYSE)
    },
    "AMERICA": {
        "Canada": "XTSE",        # Toronto Stock Exchange
        "Mexico": "XMEX",        # Mexican Stock Exchange (BMV)
        "Brazil": "BVMF",        # B3 - Brasil Bolsa Balcão
        "Chile": "XSGO",         # Santiago Stock Exchange
        "Argentina": "XBUE"      # Buenos Aires Stock Exchange
    },
    "EU": {
        "Germany": "XFRA",       # Frankfurt Stock Exchange (XETRA)
        "United Kingdom": "XLON",# London Stock Exchange (LSE)
        "France": "XPAR",        # Euronext Paris
        "Spain": "XMAD",         # Bolsa de Madrid
        "Netherlands": "XAMS",   # Euronext Amsterdam
        "Sweden": "XSTO",        # Nasdaq Stockholm
        "Italy": "XMIL",         # Borsa Italiana (Milan)
        "Switzerland": "XSWX",   # SIX Swiss Exchange
        "Poland": "XWAR",        # Warsaw Stock Exchange
        "Finland": "XHEL",       # Nasdaq Helsinki
        "Denmark": "XCSE",       # Nasdaq Copenhagen
        "Ireland": "XDUB",       # Euronext Dublin
        "Belgium": "XBRU",       # Euronext Brussels
        "Austria": "XWBO",       # Wiener Börse (Vienna Stock Exchange)
        "Portugal": "XLIS",      # Euronext Lisbon
        "Greece": "XATH",        # Athens Stock Exchange
        "Luxembourg": "XLUX",    # Luxembourg Stock Exchange
        "Czech Republic": "XPRA",# Prague Stock Exchange
        "Iceland": "XICE",       # Nasdaq Iceland
        "EU": "EU",              # Placeholder for entire EU region
        "European Union": "EU",
        "Europe": "EU"
    },
    "ASIA": {
        "China": "XSHG",         # Shanghai Stock Exchange
        "Japan": "XTKS",         # Tokyo Stock Exchange
        "South Korea": "XKRX",   # Korea Exchange
        "Hong Kong": "XHKG",     # Hong Kong Stock Exchange
        "Singapore": "XSES",     # Singapore Exchange
        "Taiwan": "XTAI",        # Taiwan Stock Exchange
        "Thailand": "XBKK",      # Stock Exchange of Thailand
        "India": "XBOM",         # Bombay Stock Exchange (BSE)
        "Indonesia": "XIDX",     # Indonesia Stock Exchange
        "Philippines": "XPHS",   # Philippine Stock Exchange
        "Vietnam": "XSTC",       # Ho Chi Minh Stock Exchange (HSX)
        "Malaysia": "XKLS",      # Bursa Malaysia
        "Pakistan": "XKAR",      # Pakistan Stock Exchange
        "Cambodia": "XCSE",      # Cambodia Securities Exchange
        "Asia": "ASIA"           # Placeholder for Asia region
    },
    "MIDDLE_EAST": {
        "Saudi Arabia": "XSAU",  # Saudi Stock Exchange (Tadawul)
        "United Arab Emirates": "XDFM",  # Dubai Financial Market
        "Israel": "XTAE",        # Tel Aviv Stock Exchange
        "Qatar": "DSMD",         # Qatar Stock Exchange
        "Bahrain": "BFX",        # Bahrain Bourse
        "Oman": "MSX",           # Muscat Securities Market
        "Kuwait": "XKUW",        # Boursa Kuwait
        "Turkey": "XIST"         # Borsa Istanbul
    },
    "OTHERS": {
        "Australia": "XASX",     # Australian Securities Exchange
        "New Zealand": "XNZE",   # New Zealand Exchange
        "South Africa": "XJSE",  # Johannesburg Stock Exchange
        "Norway": "XOSL",        # Oslo Stock Exchange
        "Russia": "MISX"         # Moscow Exchange (MOEX)
    },
    "GLOBAL": {
        "Global": "GLOBAL"       # Placeholder for global markets
    }
}


pandas_exchange_calendars = {
    "US_AMERICA": {
        "United States": "NYSE"  # New York Stock Exchange (NYSE)
    },
    "AMERICA": {
        "Canada": "TSX",         # Toronto Stock Exchange
        "Mexico": "XMEX",        # Mexican Stock Exchange (BMV)
        "Brazil": "BVMF",        # B3 - Brasil Bolsa Balcão
        "Chile": "XSGO",         # Santiago Stock Exchange
        "Argentina": "XBUE"      # Buenos Aires Stock Exchange
    },
    "EU": {
        "Germany": "XETR",       # Frankfurt Stock Exchange (XETRA)
        "United Kingdom": "XLON", # London Stock Exchange (LSE)
        "France": "XPAR",        # Euronext Paris
        "Spain": "XMAD",         # Bolsa de Madrid
        "Netherlands": "XAMS",   # Euronext Amsterdam
        "Sweden": "XSTO",        # Nasdaq Stockholm
        "Italy": "XMIL",         # Borsa Italiana (Milan)
        "Switzerland": "XSWX",   # SIX Swiss Exchange
        "Poland": "XWAR",        # Warsaw Stock Exchange
        "Finland": "XHEL",       # Nasdaq Helsinki
        "Denmark": "XCSE",       # Nasdaq Copenhagen
        "Ireland": "XDUB",       # Euronext Dublin
        "Belgium": "XBRU",       # Euronext Brussels
        "Austria": "XWBO",       # Wiener Börse (Vienna Stock Exchange)
        "Portugal": "XLIS",      # Euronext Lisbon
        "Greece": "XATH",        # Athens Stock Exchange
        "Luxembourg": "XLUX",    # Luxembourg Stock Exchange
        "Czech Republic": "XPRA", # Prague Stock Exchange
        "Iceland": "XICE",       # Nasdaq Iceland
        "EU": "EUREX",           # Placeholder for entire EU region
        "European Union": "EUREX",
        "Europe": "EUREX"
    },
    "ASIA": {
        "China": "XSHG",         # Shanghai Stock Exchange
        "Japan": "XTKS",         # Tokyo Stock Exchange
        "South Korea": "XKRX",   # Korea Exchange
        "Hong Kong": "XHKG",     # Hong Kong Stock Exchange
        "Singapore": "XSES",     # Singapore Exchange
        "Taiwan": "XTAI",        # Taiwan Stock Exchange
        "Thailand": "XBKK",      # Stock Exchange of Thailand
        "India": "XBOM",         # National Stock Exchange of India
        "Indonesia": "XIDX",     # Indonesia Stock Exchange
        "Philippines": "XPHS",   # Philippine Stock Exchange
        "Vietnam": "XHOSE",      # Ho Chi Minh Stock Exchange (HSX)
        "Malaysia": "XKLS",      # Bursa Malaysia
        "Pakistan": "XKAR",      # Pakistan Stock Exchange
        "Cambodia": "XCSX",      # Cambodia Securities Exchange
        "Asia": "XASX"           # Placeholder for entire Asia region
    },
    "MIDDLE_EAST": {
        "Saudi Arabia": "XSAU",  # Saudi Stock Exchange (Tadawul)
        "United Arab Emirates": "XDFM",  # Dubai Financial Market
        "Israel": "XTASE",       # Tel Aviv Stock Exchange
        "Qatar": "XQSE",         # Qatar Stock Exchange
        "Bahrain": "XBAH",       # Bahrain Bourse
        "Oman": "XMSM",          # Muscat Securities Market
        "Kuwait": "XKFE",        # Boursa Kuwait
        "Turkey": "XIST"         # Borsa Istanbul
    },
    "OTHERS": {
        "Australia": "XASX",     # Australian Securities Exchange
        "New Zealand": "XNZE",   # New Zealand Exchange
        "South Africa": "XJSE",  # Johannesburg Stock Exchange
        "Norway": "XOSL",        # Oslo Stock Exchange
        "Russia": "XMOS"         # Moscow Exchange (MOEX)
    },
    "GLOBAL": {
        "Global": "GLOBAL"       # Placeholder for global markets
    }
}