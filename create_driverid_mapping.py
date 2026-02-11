"""Create mapping between short and full driverIds"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

#Load f1db drivers JSON
drivers_json = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
print(f"f1db drivers: {len(drivers_json)} rows")
print(f"Sample full driverIds: {drivers_json['id'].head(10).tolist()}")

# Load qualifying CSV
qualifying_csv = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
csv_drivers = qualifying_csv['driverId'].dropna().unique()
print(f"\nCSV has {len(csv_drivers)} unique driverIds")
print(f"Sample short driverIds: {csv_drivers[:10].tolist()}")

# Try to find mapping patterns
# Short form 'hamilton' → Full form 'lewis-hamilton'
# Short form 'vettel' → Full form 'sebastian-vettel'
# Handle underscores: 'max_verstappen' → 'max-verstappen'

# Manual mappings for edge cases
manual_mappings = {
    'sainz': 'carlos-sainz-jr',
}

# Check if short forms are suffixes of full forms
mapping = {}
unmapped = []

for short_id in csv_drivers:
    # Check manual mappings first
    if short_id in manual_mappings:
        mapping[short_id] = manual_mappings[short_id]
        continue
    
    # Normalize: replace underscores with hyphens
    normalized_short = short_id.replace('_', '-')
    
    # First try: exact match (CSV might already have full IDs)
    exact_match = drivers_json[drivers_json['id'] == normalized_short]
    if len(exact_match) == 1:
        mapping[short_id] = exact_match.iloc[0]['id']
        continue
    
    # Second try: find full IDs that end with the short ID
    matches = drivers_json[drivers_json['id'].str.endswith(normalized_short)]
    
    if len(matches) == 1:
        mapping[short_id] = matches.iloc[0]['id']
    elif len(matches) > 1:
        # Multiple matches - prefer most recent driver (highest totalRaceEntries)
        if 'totalRaceEntries' in matches.columns:
            best_match = matches.nlargest(1, 'totalRaceEntries')
            mapping[short_id] = best_match.iloc[0]['id']
            print(f"Multiple matches for '{short_id}', chose: {best_match.iloc[0]['id']} (most races)")
        else:
            # Just take first match
            mapping[short_id] = matches.iloc[0]['id']
            print(f"Multiple matches for '{short_id}', chose first: {matches.iloc[0]['id']}")
    else:
        # No suffix match - try exact match on lastName
        unmapped.append(short_id)
        print(f"No match found for '{short_id}'")

print(f"\nSuccessfully mapped: {len(mapping)} / {len(csv_drivers)}")
print(f"Sample mappings:")
for short, full in list(mapping.items())[:10]:
    print(f"  {short:15s} → {full}")

# Save mapping for use in generator
mapping_df = pd.DataFrame({'short_id': list(mapping.keys()), 'full_id': list(mapping.values())})
mapping_df.to_csv(path.join(DATA_DIR, 'driverId_mapping.csv'), sep='\t', index=False)
print(f"\nSaved mapping to data_files/driverId_mapping.csv")
