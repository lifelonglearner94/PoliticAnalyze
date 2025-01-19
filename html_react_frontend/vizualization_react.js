import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PoliticalPartyComparison = () => {
  // Data transformation for fact checks (ordered by False percentage)
  const factCheckData = [
    {
      party: 'AFD',
      'True': 32.89,
      'Partially true': 3.29,
      'False': 63.82
    },
    {
      party: 'CDU',
      'True': 48.28,
      'Partially true': 7.76,
      'False': 43.97
    },
    {
      party: 'SPD',
      'True': 65.57,
      'Partially true': 9.02,
      'False': 25.41
    },
    {
      party: 'GRUENE',
      'True': 72.34,
      'Partially true': 7.80,
      'False': 19.86
    }
  ].sort((a, b) => b.False - a.False);

  // Data transformation for sentiment analysis
  const sentimentData = [
    {
      party: 'AFD',
      negative: 6.63,
      neutral: 88.79,
      positive: 4.57
    },
    {
      party: 'CDU',
      negative: 4.70,
      neutral: 92.60,
      positive: 2.70
    },
    {
      party: 'GRUENE',
      negative: 2.87,
      neutral: 93.56,
      positive: 3.56
    },
    {
      party: 'SPD',
      negative: 1.49,
      neutral: 96.97,
      positive: 1.54
    }
  ];

  // Word categories data
  const wordCategoriesData = {
    'AFD': {
      ADJ: [
        { word: 'deutsch', count: 107 },
        { word: 'eichhorster', count: 85 },
        { word: 'europäisch', count: 45 },
        { word: 'hoch', count: 37 },
        { word: 'staatlich', count: 34 }
      ],
      NOUN: [
        { word: 'partei', count: 100 },
        { word: 'alternative', count: 87 },
        { word: 'bundesgeschäftsstelle', count: 83 },
        { word: 'afd', count: 66 },
        { word: 'kind', count: 58 }
      ],
      VERB: [
        { word: 'fordern', count: 61 },
        { word: 'lehnen', count: 45 },
        { word: 'setzen', count: 38 },
        { word: 'stehen', count: 29 },
        { word: 'stellen', count: 28 }
      ]
    },
    'CDU': {
      ADJ: [
        { word: 'europäisch', count: 62 },
        { word: 'stark', count: 39 },
        { word: 'deutsch', count: 38 },
        { word: 'digital', count: 35 },
        { word: 'hoch', count: 27 }
      ],
      NOUN: [
        { word: 'land', count: 123 },
        { word: 'mensch', count: 63 },
        { word: 'kind', count: 44 },
        { word: 'sicherheit', count: 44 },
        { word: 'staat', count: 44 }
      ],
      VERB: [
        { word: 'stärken', count: 112 },
        { word: 'setzen', count: 110 },
        { word: 'schaffen', count: 68 },
        { word: 'brauchen', count: 64 },
        { word: 'unterstützen', count: 63 }
      ]
    },
    'GRUENE': {
      ADJ: [
        { word: 'europäisch', count: 96 },
        { word: 'stark', count: 68 },
        { word: 'gemeinsam', count: 43 },
        { word: 'international', count: 42 },
        { word: 'sozial', count: 36 }
      ],
      NOUN: [
        { word: 'mensch', count: 187 },
        { word: 'land', count: 126 },
        { word: 'sicherheit', count: 55 },
        { word: 'gesellschaft', count: 53 },
        { word: 'demokratie', count: 47 }
      ],
      VERB: [
        { word: 'stärken', count: 106 },
        { word: 'brauchen', count: 79 },
        { word: 'setzen', count: 75 },
        { word: 'unterstützen', count: 72 },
        { word: 'schaffen', count: 53 }
      ]
    },
    'SPD': {
      ADJ: [
        { word: 'europäisch', count: 80 },
        { word: 'sozial', count: 46 },
        { word: 'wichtig', count: 40 },
        { word: 'öffentlich', count: 38 },
        { word: 'stark', count: 33 }
      ],
      NOUN: [
        { word: 'land', count: 85 },
        { word: 'mensch', count: 83 },
        { word: 'arbeit', count: 46 },
        { word: 'familie', count: 42 },
        { word: 'sicherheit', count: 41 }
      ],
      VERB: [
        { word: 'stärken', count: 98 },
        { word: 'setzen', count: 93 },
        { word: 'schaffen', count: 68 },
        { word: 'unterstützen', count: 67 },
        { word: 'sorgen', count: 54 }
      ]
    }
  };

  const colors = {
    True: '#4CAF50',
    'Partially true': '#FFC107',
    False: '#F44336',
    negative: '#FF9999',
    neutral: '#AAAAAA',
    positive: '#99FF99'
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Fact Check Analysis by Party (Ordered by False Statements)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={factCheckData}>
                <XAxis dataKey="party" />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 20, 40, 60, 80, 100]}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip formatter={(value) => [`${value}%`]} />
                <Legend />
                <Bar dataKey="True" stackId="a" fill={colors.True} />
                <Bar dataKey="Partially true" stackId="a" fill={colors['Partially true']} />
                <Bar dataKey="False" stackId="a" fill={colors.False} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Sentiment Analysis by Party</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentData}>
                <XAxis dataKey="party" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="positive" stackId="a" fill={colors.positive} />
                <Bar dataKey="neutral" stackId="a" fill={colors.neutral} />
                <Bar dataKey="negative" stackId="a" fill={colors.negative} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Most Frequently Used Words by Category and Party</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(wordCategoriesData).map(([party, categories]) => (
              <div key={party} className="border rounded-lg p-4">
                <h3 className="font-bold mb-4 text-lg">{party}</h3>
                <div className="space-y-4">
                  {Object.entries(categories).map(([category, words]) => (
                    <div key={category} className="space-y-2">
                      <h4 className="font-semibold">{category}</h4>
                      <ul className="list-disc list-inside">
                        {words.map((item, index) => (
                          <li key={index} className="text-sm">
                            {item.word} ({item.count})
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PoliticalPartyComparison;
