//https://claude.site/artifacts/5fe077cf-114f-495d-bc72-228386daab7c


import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PoliticalPartyComparison = () => {
  // Data transformation for fact checks (ordered by False percentage)
  const factCheckData = [
    {
      party: 'AfD',
      'Wahr': 32.89,
      'Teilweise wahr': 3.29,
      'Falsch': 63.82
    },
    {
      party: 'CDU',
      'Wahr': 48.28,
      'Teilweise wahr': 7.76,
      'Falsch': 43.97
    },
    {
      party: 'SPD',
      'Wahr': 65.57,
      'Teilweise wahr': 9.02,
      'Falsch': 25.41
    },
    {
      party: 'GRÜNE',
      'Wahr': 72.34,
      'Teilweise wahr': 7.80,
      'Falsch': 19.86
    }
  ].sort((a, b) => b.Falsch - a.Falsch);

  // Data transformation for sentiment analysis
  const sentimentData = [
    {
      party: 'AfD',
      negativ: 6.63,
      neutral: 88.79,
      positiv: 4.57
    },
    {
      party: 'CDU',
      negativ: 4.70,
      neutral: 92.60,
      positiv: 2.70
    },
    {
      party: 'GRÜNE',
      negativ: 2.87,
      neutral: 93.56,
      positiv: 3.56
    },
    {
      party: 'SPD',
      negativ: 1.49,
      neutral: 96.97,
      positiv: 1.54
    }
  ];

  const wordCategories = ['Adjektive', 'Substantive', 'Verben'];
  const parties = ['AfD', 'CDU', 'GRÜNE', 'SPD'];

  const wordCategoriesData = {
    'AfD': {
      'Adjektive': [
        { word: 'deutsch', count: 107 },
        { word: 'europäisch', count: 45 },
        { word: 'hoch', count: 37 },
        { word: 'staatlich', count: 34 },
        { word: 'politisch', count: 25 }
      ],
      'Substantive': [
        { word: 'partei', count: 100 },
        { word: 'alternative', count: 87 },
        { word: 'bundesgeschäftsstelle', count: 83 },
        { word: 'afd', count: 66 },
        { word: 'kind', count: 58 }
      ],
      'Verben': [
        { word: 'fordern', count: 61 },
        { word: 'lehnen', count: 45 },
        { word: 'setzen', count: 38 },
        { word: 'stehen', count: 29 },
        { word: 'stellen', count: 28 }
      ]
    },
    'CDU': {
      'Adjektive': [
        { word: 'europäisch', count: 62 },
        { word: 'stark', count: 39 },
        { word: 'deutsch', count: 38 },
        { word: 'digital', count: 35 },
        { word: 'hoch', count: 27 }
      ],
      'Substantive': [
        { word: 'land', count: 123 },
        { word: 'mensch', count: 63 },
        { word: 'kind', count: 44 },
        { word: 'sicherheit', count: 44 },
        { word: 'staat', count: 44 }
      ],
      'Verben': [
        { word: 'stärken', count: 112 },
        { word: 'setzen', count: 110 },
        { word: 'schaffen', count: 68 },
        { word: 'brauchen', count: 64 },
        { word: 'unterstützen', count: 63 }
      ]
    },
    'GRÜNE': {
      'Adjektive': [
        { word: 'europäisch', count: 96 },
        { word: 'stark', count: 68 },
        { word: 'gemeinsam', count: 43 },
        { word: 'international', count: 42 },
        { word: 'sozial', count: 36 }
      ],
      'Substantive': [
        { word: 'mensch', count: 187 },
        { word: 'land', count: 126 },
        { word: 'sicherheit', count: 55 },
        { word: 'gesellschaft', count: 53 },
        { word: 'demokratie', count: 47 }
      ],
      'Verben': [
        { word: 'stärken', count: 106 },
        { word: 'brauchen', count: 79 },
        { word: 'setzen', count: 75 },
        { word: 'unterstützen', count: 72 },
        { word: 'schaffen', count: 53 }
      ]
    },
    'SPD': {
      'Adjektive': [
        { word: 'europäisch', count: 80 },
        { word: 'sozial', count: 46 },
        { word: 'wichtig', count: 40 },
        { word: 'öffentlich', count: 38 },
        { word: 'stark', count: 33 }
      ],
      'Substantive': [
        { word: 'land', count: 85 },
        { word: 'mensch', count: 83 },
        { word: 'arbeit', count: 46 },
        { word: 'familie', count: 42 },
        { word: 'sicherheit', count: 41 }
      ],
      'Verben': [
        { word: 'stärken', count: 98 },
        { word: 'setzen', count: 93 },
        { word: 'schaffen', count: 68 },
        { word: 'unterstützen', count: 67 },
        { word: 'sorgen', count: 54 }
      ]
    }
  };

  const colors = {
    Wahr: '#4CAF50',
    'Teilweise wahr': '#FFC107',
    Falsch: '#F44336',
    negativ: '#FF9999',
    neutral: '#AAAAAA',
    positiv: '#99FF99'
  };

  return (
    <div className="w-full space-y-6 p-4 bg-gray-50">
      <h1 className="text-2xl font-bold text-center mb-8">Analyse der Parteiprogramme</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Fact Check Analysis */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>KI-Faktenüberprüfung aller Behauptungen in den Wahlprogrammen</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
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
                  <Bar dataKey="Wahr" stackId="a" fill={colors.Wahr} />
                  <Bar dataKey="Teilweise wahr" stackId="a" fill={colors['Teilweise wahr']} />
                  <Bar dataKey="Falsch" stackId="a" fill={colors.Falsch} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Sentiment Analysis */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Stimmungsanalyse nach Partei</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sentimentData}>
                  <XAxis dataKey="party" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="positiv" stackId="a" fill={colors.positiv} />
                  <Bar dataKey="neutral" stackId="a" fill={colors.neutral} />
                  <Bar dataKey="negativ" stackId="a" fill={colors.negativ} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Word Analysis Tables */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Vergleich der häufigsten Wörter nach Kategorie</CardTitle>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {wordCategories.map((category) => (
              <div key={category} className="mb-8">
                <h3 className="font-bold mb-4 text-lg">{category}</h3>
                <table className="w-full border-collapse bg-white">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border p-2 text-left w-1/4">AfD</th>
                      <th className="border p-2 text-left w-1/4">CDU</th>
                      <th className="border p-2 text-left w-1/4">GRÜNE</th>
                      <th className="border p-2 text-left w-1/4">SPD</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[0, 1, 2, 3, 4].map((index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        {parties.map((party) => (
                          <td key={party} className="border p-2 text-sm">
                            {wordCategoriesData[party][category][index] ? (
                              <span>
                                {wordCategoriesData[party][category][index].word}
                                <span className="text-gray-500 ml-1">
                                  ({wordCategoriesData[party][category][index].count})
                                </span>
                              </span>
                            ) : '-'}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default PoliticalPartyComparison;
