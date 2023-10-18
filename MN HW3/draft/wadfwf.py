size = 248


for y in range(size):
    for x in range(size):
        amplitude = 1.0
        frequency = 1.0
        noiseHeight = 0

        for i in range(data.octaves):
            sample = Vector2(x / data.noiseScale * frequency + octaveOffsets[i].x,
                             y / data.noiseScale * frequency + octaveOffsets[i].y)
            perlinValue = Mathf.PerlinNoise(sample.x, sample.y)

            noiseHeight += perlinValue * amplitude
            if data.persistenceType == PersistenceType.Quarter:
                amplitude *= 1 / 4.0
            elif data.persistenceType == PersistenceType.Half:
                amplitude *= .5
            elif data.persistenceType == PersistenceType.Sqrt:
                amplitude *= 1 / math.sqrt(2)
            elif data.persistenceType == PersistenceType.Default:
                amplitude *= 1
            else:
                raise ValueError('Invalid persistence type')

            frequency *= data.lacunarity

        if noiseHeight > maxNoiseHeight:
            maxNoiseHeight = noiseHeight
        elif noiseHeight < minNoiseHeight:
            minNoiseHeight = noiseHeight

        noiseMap[x][y] = noiseHeight

