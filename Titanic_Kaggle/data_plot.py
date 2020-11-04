import matplotlib.pyplot as plt
import seaborn as sns


def missing_data(train_data):
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    fig1.subplots_adjust(bottom=0.25, wspace=0.25, left=0.05, right=0.95)
    missing_data = sns.heatmap(train_data.isnull(), cmap='Greys_r',
                               yticklabels=False, cbar=False, ax=ax1)
    for i in range(len(train_data['PassengerId'])):
        missing_data.axvline(i, color='white', lw=1)

    # sns.boxplot(x='Pclass', y='Age', data=train_data, palette='winter', ax=ax1[1])

    # sns.violinplot(x='Pclass', y='Age', data=train_data,
    #                hue='Sex', split=True, ax=ax1[1])
    # fig1.canvas.set_window_title('Data Cleaning')
    #
    # # Add a horizontal grid to the plot
    # ax1[1].yaxis.grid(True, linestyle='-', which='major',
    #                   color='lightgrey', alpha=0.5)
    # ax1[1].set_axisbelow(True)
    # ax1[1].legend(loc=(.55, 0.82))
    ax1.set_title('Missing data shown in white')
    # ax1[1].set_title('Age distribution based on Class and Gender')

    plt.show()


def data_explore(train_data):
    fig2, ax2 = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))
    fig2.subplots_adjust(bottom=0.08, top=0.95, left=0.1, right=0.98,
                         wspace=0.25, hspace=0.35)
    fig2.canvas.set_window_title('Titanic Dataset')
    sns.countplot(x='Survived', data=train_data, hue='Sex', ax=ax2[0, 0])

    sns.violinplot(x='Pclass', y='Age', data=train_data,
                   hue='Sex', split=True, ax=ax2[0, 1])
    fig2.canvas.set_window_title('Data Cleaning')

    sns.countplot(x=train_data[train_data['Sex'] == 'male']['Survived'],
                  hue=train_data['Pclass'],
                  ax=ax2[1, 0])

    sns.countplot(x=train_data[train_data['Sex'] == 'female']['Survived'],
                  hue=train_data['Pclass'],
                  ax=ax2[1, 1])

    # Add horizontal grid to the plots
    for i, _ in enumerate(ax2):
        for j, _ in enumerate(ax2):
            ax2[i, j].yaxis.grid(True, linestyle='-', which='major',
                                 color='lightgrey', alpha=0.7)
            # Hide these grid behind plot objects
            ax2[i, j].set_axisbelow(True)

    ax2[1, 0].legend(['Fist Class', 'Second Class', 'Third Class'])
    ax2[1, 1].legend(['Fist Class', 'Second Class', 'Third Class'])
    ax2[1, 0].set_title('Men')
    ax2[1, 1].set_title('Women')
    ax2[0, 1].legend(loc=(.53, 0.75))
    ax2[0, 1].set_title('Age distribution based on Class and Gender')
    plt.show()

# fig = sns.PairGrid(train_data)
# sns.regplot(x, y, ax=ax1)
# sns.kdeplot(x, ax=ax2)
# sns.rugplot(train_data['Age'])
# sns.stripplot(x='Age', y='PassengerId', data=train_data, jitter=False)

# sns.pairplot(train_data)
# sns.barplot(train_data['Pclass'], train_data['Survived'])
# fig2.show()
