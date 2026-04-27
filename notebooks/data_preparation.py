import pandas as pd
from sklearn.model_selection import train_test_split

def data_preparation(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["CustomerID", "AdvertisingPlatform", "AdvertisingTool"])

    FEATURES = [
        'CampaignType', 'AdSpend', 'ClickThroughRate', 'WebsiteVisits', 
        'PagesPerVisit', 'TimeOnSite', 'EmailOpens', 'EmailClicks', 
        'PreviousPurchases', 'LoyaltyPoints'
    ]
    TARGET = 'Conversion'
    
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    # Split dataset into training and testing datasets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Split training dataset into training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test