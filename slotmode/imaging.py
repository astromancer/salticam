import matplotlib.pyplot as plt


def display_slot_image(image_stack, channels=()):
    """
    Image of the slot across all 4 amplifier channels

    Parameters
    ----------
    image_stack
    channels

    Returns
    -------

    """
    assert len(image_stack) == 4, 'This function requires images from all 4 ' \
                                  'amplifier channels'

    if isinstance(channels, int):
        channels = channels,

    #
    fig, axes = plt.subplots(1, 4,
                             figsize=(23, 1),  # choose based on data shape
                             sharey='all',  # ,
                             gridspec_kw=dict(wspace=0.005,
                                              top=0.8,
                                              bottom=0.13,
                                              left=0.015,
                                              right=0.995, ))
    for i, ax in enumerate(axes):
        # setting aspect auto will stretch the image, but we don't care since
        #  this is meant for quick look only
        ax.imshow(image_stack[i], origin='lower', aspect='auto')
        # todo: better colour scale!!

        # highlight extraction channel
        if i in channels:
            for pos, spine in ax.spines.items():
                spine.set_color('limegreen')
                spine.set_linewidth(2.5)

    # fig.tight_layout()
    fig.suptitle('SlotMode Image (4 channels)', y=1, fontweight='bold', )
    # va='bottom')

    return fig
