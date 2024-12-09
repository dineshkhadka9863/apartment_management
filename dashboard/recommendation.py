
def calculate_similarity(apartment1, apartment2):
    # Calculate similarity score based on attribute values
    similarity_score = 0
    if apartment1['location'] == apartment2['location']:
        similarity_score += 1
    if apartment1['floor'] == apartment2['floor']:
        similarity_score += 1
    if apartment1['bhk'] == apartment2['bhk']:
        similarity_score += 1
    if apartment1['parking'] == apartment2['parking']:
        similarity_score += 1
    if apartment1['ac'] == apartment2['ac']:
        similarity_score += 1
    if apartment1['swimming_pool'] == apartment2['swimming_pool']:
        similarity_score += 1
    if apartment1['wifi'] == apartment2['wifi']:
        similarity_score += 1
    return similarity_score

def get_recommended_apartments(viewed_apartment, all_apartments):
    similar_apartments = []
    for apartment in all_apartments:
        if apartment['id'] != viewed_apartment['id']:
            similarity = calculate_similarity(viewed_apartment, apartment)
            similar_apartments.append((apartment, similarity))

    similar_apartments.sort(key=lambda x: x[1], reverse=True)
    return similar_apartments
