def merge_predictions_with_tanimoto(predictions_path, tanimoto_path, output_path):
    import pandas as pd
    sheet_one_df = pd.read_excel(predictions_path)
    sheet_two_df = pd.read_excel(tanimoto_path)
    merged_df = pd.merge(
        sheet_one_df,
        sheet_two_df[['SMILES', 'Average_Tanimoto_Distance']],
        on='SMILES',
        how='left'
    )
    merged_df.to_excel(output_path, index=False)
    return merged_df