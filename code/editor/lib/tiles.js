'use server'

export const loadBaseImage = async (name) => {
    let url = new URL(`/base/${name}`, process.env.API_URL);
    const response = await fetch(url)

    return await response.blob()
}


export const loadSegmentation = async (name) => {
    let url = new URL(`/seg/${name}`, process.env.API_URL);
    const response = await fetch(url)

    return await response.json()
}

export const generateImage = async (position, grid) => {
    let url = new URL(`/eval`, process.env.API_URL);

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "segmentation": grid,
            "base": position
        })
    })

    if(!response.ok) {
        console.error(response)
        return ""
    }

    const data = await response.json()
    return data
}

export const reGenerateImage = async (position, grid, imageBase64) => {
    let url = new URL(`/reeval`, process.env.API_URL);

    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "image": imageBase64,
            "segmentation": grid,
            "base": position
        })
    })

    if(!response.ok) {
        console.error(response)
        return ""
    }

    const data = await response.json()
    return data
}