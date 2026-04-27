def get_label(error):
    label = " Anomaly detected" if error > threshold else " Normal structure"
    return label