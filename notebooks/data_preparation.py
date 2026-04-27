import pandas as pd
from sklearn.model_selection import train_test_split

def data_preparation(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Filter irrevalent variables
    df = df.drop(columns=["CustomerID", "AdvertisingPlatform", "AdvertisingTool"])

    FEATURES = [
        "CampaignType", "AdSpend", "ClickThroughRate", "WebsiteVisits", 
        "PagesPerVisit", "TimeOnSite", "EmailOpens", "EmailClicks", 
        "PreviousPurchases", "LoyaltyPoints"
    ]
    TARGET = "Conversion"
    
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Split train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test