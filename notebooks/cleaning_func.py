def pumpit_clean(csv):
    # ``` This function requires: numpy, pandas, os, 
    # ```

# Build path to data folder
    # try:
    df = pd.read_csv(csv)
    # except:
    #     print('Please input a CSV for cleaning, including path and as a string')
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

# Setting 0 to NaN in most numerical variables
    #cap the 'amount_tsh' at 1500
    df['amount_tsh_capped'] = df['amount_tsh'].apply(lambda x: min(x, 15000))
    # might want to replace this with bins
    scaler = MinMaxScaler() 
    df['amount_tsh_scaled'] = scaler.fit_transform(df[['amount_tsh_capped']])
    # gps_height 
    df['gps_height'] = df['gps_height'].replace(0, np.nan)

    # location:
    df['longitude'] = df['longitude'].replace(0, np.nan)
    df['latitude'] = df['latitude'].where(df['latitude'] < -0.5, np.nan) # too close to the equator

    # Fill population using median by district_code
    df['population'] = df.groupby('district_code')['population'].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill any still missing with median by region
    df['population'] = df.groupby('region')['population'].transform(
        lambda x: x.fillna(x.median())
    )


    df['construction_year'] = df['construction_year'].replace(0, np.nan) 

    ### Encode categorical variables
    from sklearn.preprocessing import LabelEncoder

    # Selected categorical variable 
    catvars = [
    'region','quantity','management','water_quality',
    'quantity','payment','source','basin',
    ]

    #  Drop high-cardinality columns
    drop_cols = ['funder', 'installer', 'wpt_name', 'subvillage', 'scheme_name']
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in drop_cols and col != 'id']

    # One-hot encode medium-cardinality features (between 3 and 10 unique values)
    onehot_cols = [col for col in categorical_cols if 2 < df[col].nunique() <= 10]
    df_encoded = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    #  Label encode low-cardinality features (<= 2 unique values)
    label_enc_cols = [col for col in categorical_cols if df[col].nunique() <= 2]
    label_encoders = {}
    for col in label_enc_cols:
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    return df 