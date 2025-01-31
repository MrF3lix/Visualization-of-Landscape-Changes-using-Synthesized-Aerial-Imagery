'use server'

export const loadLocations = async (searchQuery) => {

    let url = new URL('/search/geocode/v6/forward', process.env.MAPBOX_API_URL);

    url.search = new URLSearchParams({
        q: searchQuery,
        country: 'ch',
        access_token: process.env.MAPBOX_API_TOKEN,
        limit: 5
    });

    const response = await fetch(url)
    const payload = await response.json()

    if(payload?.type && payload.type == 'FeatureCollection') {
        return payload.features.map(feature => {

            return {
                value: feature,
                label: `${feature.properties.full_address}`
            }
        })
    }
    return []
}
