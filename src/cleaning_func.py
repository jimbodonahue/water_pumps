# create a class that we can pass to the pipeline
# run this cell to initiate the function, then the pipeline *should* run smoothly
def clean_func(df):
    #
    # Construction year
    df['construction_year'] = df['construction_year'].replace(0, np.nan)
    #Impute using region + installer
    df['construction_year'] = df.groupby(['region', 'installer'])['construction_year'].transform(
        lambda x: x.fillna(x.median())
    )
    #Impute using region only (for rows still missing)
    df['construction_year'] = df.groupby('region')['construction_year'].transform(
        lambda x: x.fillna(x.median())
    )
    #Use recorded year - 13
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df['recorded_year'] = df['date_recorded'].dt.year
    df['construction_year'] = df['construction_year'].fillna(df['recorded_year'] - 13)
    #
    # gps_height
    df['gps_height'] = df['gps_height'].apply(lambda x: np.nan if x <= 0 else x)
    # Fill using median per lga
    df['gps_height'] = df.groupby('lga')['gps_height'].fillna(df['gps_height'].median())
    # Fill any still missing using region median
    df['gps_height'] = df.groupby(['region'])['gps_height'].fillna(df['gps_height'].median())
    # Longitude and latitude
    df['longitude'] = df['longitude'].replace(0, np.nan)
    df['latitude'] = df['latitude'].where(df['latitude'] < -0.5, np.nan) # too close to the equator
    for i in ['latitude','longitude']: # loop to fill by lga, region
        df[i] = df.groupby(['lga'])[i].fillna(df[i].median())
        df[i] = df.groupby(['region'])[i].fillna(df[i].median())
    # population
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
    df['population'] = pd.cut(df['population'], [-1,1,25,90,160,260,9999999], labels=[0,0.2,0.3,0.4,0.6,1])
    df['population'] = df['population'].astype(float)
    #
    # amount_tsh
    df['amount_tsh'] = df['amount_tsh'].apply(lambda x: min(x, 15000))
    df['amount_tsh'] = df['amount_tsh'].apply(lambda x: np.power(x,0.3))
        ### Encode categorical variables
    # Encode 'quantity' (and typo fix: 'insufficent' -> 'insufficient')
    df['quantity'] = df['quantity'].replace({
        'enough': 1,
        'seasonal': 0.6,
        'insufficient': 0.4,
        'dry': 0,
        'unknown': 0
    })
    df.quantity = pd.to_numeric(df.quantity, errors='coerce')

    # Encode 'water_quality' as binary: good = 1, else 0
    df['water_quality'] = np.where(df['water_quality'] == 'soft', 1, 0)
    # Encode 'waterpoint_type' (1 = preferred type, 0 = everything else)
    preferred_waterpoint = ['communal standpipe multiple', 'communal standpipe']
    df['waterpoint_type'] = df['waterpoint_type'].apply(lambda x: 1 if x in preferred_waterpoint else 0)
    # Encode 'permit' as binary: True = 1, False, missing = 0
    df['permit'] = np.where(df['permit'] == 'True', 1, 0)
    # Encode 'payment' as binary: never pay = 0, else = 1
    df['payment'] = np.where(df['payment'] == 'never pay', 0, 1)
    # Encode 'source' (1 = preferred sources, 0 = everything else)
    preferred_sources = ['spring', 'river', 'rainwater harvesting']
    df['source'] = df['source'].apply(lambda x: 1 if x in preferred_sources else 0)
    # Encode 'payment' as binary: never pay = 0, else = 1
    df['extraction_type_class'] = np.where(df['extraction_type_class'] == 'gravity', 0, 1)
    # Encode 'scheme_management' (1 = VWC, others 0)
    df['scheme_management'] = np.where(df['scheme_management'] == 'VWC', 0, 1)
    # one hot encoder for basin
    df = pd.get_dummies(data=df, columns=['basin'], drop_first=True, dtype=int)
    ### Select what's good
     #  Drop other columns and only keep these:
    # df_small = df[['amount_tsh',
    #     'gps_height',
    #     'population',
    #     'construction_year',
    #     'extraction_type_class',
    #     'payment',
    #     'water_quality',
    #     'quantity',
    #     'source',
    #     'waterpoint_type'
    #    ]]
    #  #  Drop other columns and only keep these:
    # df_medium = df[['amount_tsh',
    #          'gps_height',
    #          'longitude',
    #          'latitude',
    #          'population',
    #          'construction_year',
    #          'extraction_type_class',
    #          'payment',
    #         'water_quality',
    #         'quantity',
    #         'source',
    #         'waterpoint_type',, 'basin_Lake Nyasa', 'basin_Lake Rukwa',
    #         'basin_Lake Tanganyika', 'basin_Lake Victoria', 'basin_Pangani',
    #         'basin_Rufiji', 'basin_Ruvuma / Southern Coast', 'basin_Wami / Ruvu'
    #         'scheme_management'
    #        ]]
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
            'scheme_management', 'basin_Lake Nyasa', 'basin_Lake Rukwa',
            'basin_Lake Tanganyika', 'basin_Lake Victoria', 'basin_Pangani',
            'basin_Rufiji', 'basin_Ruvuma / Southern Coast', 'basin_Wami / Ruvu'
           ]]
    df['tshXpayment'] = df.amount_tsh * df.payment
    df['extractXsource'] = df.extraction_type_class * df.source
    df['popXtsh'] = df.population * df.amount_tsh
    df['popXquant'] = df.population * df.quantity
    df['popXsource'] = df.population * df.source
    df['extractXheight'] = df.extraction_type_class * df.gps_height
    df['typeXsource'] = df.waterpoint_type * df.source
    df['typeXyear'] = df.waterpoint_type * df.construction_year
    df['yearXpop'] = df.construction_year * df.population
    df['quantXsource'] = df.quantity * df.source
    df['yearsq'] = np.sqrt(df.construction_year + 1)
    df_large = df
    return df#_small, df_medium, df_large

#def clean_data(self, df):
#    df = self.clean_numeric(df)
#    df = self.clean_categorical(df)
#    df = self.selection(df)
#    return df

print('ready for piping')

my_transformer = FunctionTransformer(clean_func)
