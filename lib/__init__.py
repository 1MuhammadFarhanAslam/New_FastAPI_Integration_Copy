# Import all submodules.
from . import protocol
from . import globals

__version__ = "3.3.6"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)


BLACKLISTED_IPS_SEG = []
BLACKLISTED_IPS = []

BLACKLISTED_MINER_COLDKEYS = ['5E2RHBCtUcbw5KTGvMWfeaUXKrmcrDVvqYb7EWmZzqfbFHBH',
    '5Dq1CKGLEdquzy4cr63y39qsmZxZ2g5rSbxnkpLrHXVwXWoV',
    '5FCZ8g13LsRdASt3rfwEfQZ9iDkDDhoUvScCSFRqocss2ukP',
    '5CyFQsNqNkGuFsWwbjS3TMW7wLtzAQBKfL4qya7HYhFX1h2s',
    '5FLF7CKccooocsgSDqDRAqFUktr9LwYK8ALbbZ2FJzmLpJXx',
    '5FcQHaUjQnJNnRH2ZM9buA4PcxYaQCx79BNSCaPVn17EztMh',
    '5FqVJrW2jCg2d8nBqrZaivy3SxE2MYUtU2o68FjFwoMdxjG4',
    '5HeP2f5mSqTJVXwYTUtt98efSGRb7JpFySdny63erdUA9FSd',
    '5FKwEVKAzVieNyajbPEiyoYGNQYwKX1sLoe3EqZK9Bu3Kcxx',
    '5G6Aq86k3vFnUBA1ChJGJypz2QprKJgmFtv1mNHavEG9XmQ7',
    '5GEz9ZQXVkAupXi7br8HtxjVCbCwEWsx6wGJkzg9mPsu9qTG',
    '5HBVrFGy6oYhhh71m9fFGYD7zbKyAeHnWN8i8s9fJTBMCtEE',
    ]

BLACKLISTED_MINER_HOTKEYS = ['5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe','5HBVrFGy6oYhhh71m9fFGYD7zbKyAeHnWN8i8s9fJTBMCtEE']

BLACKLISTED_VALIDATORS = ['5G1NjW9YhXLadMWajvTkfcJy6up3yH2q1YzMXDTi6ijanChe']
legit_validators = ['5F4tQyWrhfGVcNhoqeiNsR6KjD4wMZ2kfhLj4oHYuyHbZAc3', '5HTZipxVCMqzhLt9QKi2Nxj3Fd6TCSnzTjBKR3vtiuTkuq1B', '5DHgZowFNKcGeAiN6EyJGA5he4PZe1Jp41yt5QPaC9jF68QD',
    '5HbLYXUBy1snPR8nfioQ7GoA9x76EELzEq9j7F32vWUQHm1x', '5DqQ1r8Xr6u88QrJcfxNisvzjcGj95W7H2SzX1Nwgd1CrDnK',
    '5Hddm3iBFD2GLT5ik7LZnT3XJUnRnN8PoeCFgGQgawUVKNm8',
    '5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2',
    '5CaNj3BarTHotEK1n513aoTtFeXcjf6uvKzAyzNuv9cirUoW',
    '5EhvL1FVkQPpMjZX4MAADcW42i3xPSF1KiCpuaxTYVr28sux',
    '5HK5tp6t2S59DywmHRWPBVJeJ86T61KjurYqeooqj8sREpeN',
    '5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v',
    '5Dd8gaRNdhm1YP7G1hcB1N842ecAUQmbLjCRLqH5ycaTGrWv',
    '5DvTpiniW9s3APmHRYn8FroUWyfnLtrsid5Mtn5EwMXHN2ed',
    '5CXRfP2ekFhe62r7q3vppRajJmGhTi7vwvb2yr79jveZ282w',
    '5HNQURvmjjYhTSksi8Wfsw676b4owGwfLR2BFAQzG7H3HhYf',
    '5FKstHjZkh4v3qAMSBa1oJcHCLjxYZ8SNTSz1opTv4hR7gVB',
    '5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp',
    '5G3f8VDTT1ydirT3QffnV2TMrNMR2MkQfGUubQNqZcGSj82T',
    '5ED6jwDECEmNvSp98R2qyEUPHDv9pi14E6n3TS8CicD6YfhL',
    '5FFM6Nvvm78GqyMratgXXvjbqZPi7SHgSQ81nyS96jBuUWgt',
    '5Fq5v71D4LX8Db1xsmRSy6udQThcZ8sFDqxQFwnUZ1BuqY5A',
    '5DnXm2tBGAD57ySJv5SfpTfLcsQbSKKp6xZKFWABw3cYUgqg',
    '5HeKSHGdsRCwVgyrHchijnZJnq4wiv6GqoDLNah8R5WMfnLB',
    '5CVS9d1NcQyWKUyadLevwGxg6LgBcF9Lik6NSnbe5q59jwhE',
    '5Dz8ShM6rtPw1GBAaqxjycT9LF1TC3iDpzpUH9gKr85Nizo6',
    '5HBVrFGy6oYhhh71m9fFGYD7zbKyAeHnWN8i8s9fJTBMCtEE',
    '5GsL9zNp1CdKmKSYBGjeF9kGReRpS8KdQv2yJ3mHVwKq2YCq',]

MIN_STAKE = 0
WHITELISTED_VALIDATORS = legit_validators
