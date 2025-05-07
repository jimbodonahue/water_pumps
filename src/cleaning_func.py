def pumpit_clean(df :pd.core.frame.DataFrame):
    # ``` This function requires: numpy, pandas, os,
    # ```

    # handle extra imports here
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder

    ## Numerical Variables

    # amount_tsh: cap at 15000
    df['amount_tsh'] = df['amount_tsh'].apply(lambda x: min(x, 15000))
    # might want to replace this with bins
    scaler = MinMaxScaler()
    df['amount_tsh'] = scaler.fit_transform(df[['amount_tsh']]) # names kept consistent
    # gps_height
    # # Replace invalid gps_height values (e.g. 0 or negative)
    df['gps_height'] = df['gps_height'].apply(lambda x: np.nan if x <= 0 else x)
    # Fill using median per basin
    df['gps_height'] = df.groupby('basin')['gps_height'].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill any still missing using region median
    df['gps_height'] = df.groupby('region')['gps_height'].transform(
        lambda x: x.fillna(x.median())
    )
    df['gps_height'] = scaler.fit_transform(df[['gps_height']])

    # location:
    df['longitude'] = df['longitude'].replace(0, np.nan)
    df['latitude'] = df['latitude'].where(df['latitude'] < -0.5, np.nan) # too close to the equator
    for i in ['latitude','longitude']:
        df[i] = df.groupby('lga')[i].transform(lambda x: x.fillna(x.median))
        df[i] = df.groupby('region')[i].transform(lambda x: x.fillna(x.median))

    # Fill population using median by district_code
    df['population'] = df.groupby('lga')['population'].transform(
        lambda x: x.fillna(x.median())
    )
    # Fill any still missing with median by region, then overall median
    df['population'] = df.groupby('region')['population'].transform(
        lambda x: x.fillna(x.median())
    )
    df['population'] = df['population'].fillna(df.population.median)
    # Bin the outcome, see how it behaves
    df['population'] = pd.cut(df['population'], [-1,1,25,90,160,260,9999999], labels=[0,0.1,0.2,0.4,0.6,0.9])
    df['population'] = df['population'].astype(float)
    # Construction year ??
    df['construction_year'] = df['construction_year'].replace(0, np.nan)
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['recorded_year'] = df['date_recorded'].dt.year
    #Impute using region + installer
    df['construction_year'] = df.groupby(['region', 'installer'])['construction_year'].transform(
        lambda x: x.fillna(x.median())
    )
    #Impute using region only (for rows still missing)
    df['construction_year'] = df.groupby('region')['construction_year'].transform(
        lambda x: x.fillna(x.median())
    )
    #Use recorded year - 5
    df['construction_year'] = df['construction_year'].fillna(df['recorded_year'] - 5)
    df['construction_year'] = scaler.fit_transform(df[['construction_year']])

    ### Encode categorical variables

    # Encode 'quantity' (and typo fix: 'insufficent' -> 'insufficient')
    df['quantity'] = df['quantity'].replace({
        'enough': 1,
        'seasonal': 0.6,
        'insufficient': 0.4,
        'dry': 0,
        'unknown': 0
    })
    # Encode 'water_quality' as binary: good = 1, else 0
    df['water_quality'] = np.where(df['water_quality'] == 'soft', 1, 0)
    # Encode 'waterpoint_type' (1 = preferred type, 0 = everything else)
    preferred_waterpoint = ['hand pump', 'communal standpipe']
    df['waterpoint_type'] = df['waterpoint_type'].apply(lambda x: 1 if x in preferred_waterpoint else 0)
    # Encode 'payment' as binary: never pay = 0, else = 1
    df['payment'] = np.where(df['payment'] == 'never pay', 0, 1)
    # Encode 'source' (1 = preferred sources, 0 = everything else)
    preferred_sources = ['spring', 'river', 'rainwater harvesting']
    df['source'] = df['source'].apply(lambda x: 1 if x in preferred_sources else 0)
    # Encode 'payment' as binary: never pay = 0, else = 1
    df['extraction_type_class'] = np.where(df['extraction_type_class'] == 'gravity', 0, 1)

    #  Drop other columns and only keep these:
    df = df[['amount_tsh',
             'gps_height',
             'longitude',
             'latitude',
             'population',
             'construction_year',
             'extraction_type_class',
             'payment',
            'water_quality',
            'quantity',
            'source',
            'waterpoint_type',
       ]]

    return df
