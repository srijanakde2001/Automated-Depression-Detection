#include <bits/stdc++.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// #pragma GCC target ("avx2")
// #pragma GCC optimization ("O3")
// #pragma GCC optimization ("unroll-loops")
// #pragma optimization_level 3
// #pragma GCC optimize("Ofast,no-stack-protector,unroll-loops,fast-math,O3")
// #pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")

#define f0r(a, b) for (long long a = 0; a < (b); ++a)
#define f1r(a, b, c) for (long long a = (b); a < (c); ++a)
#define f0rd(a, b) for (long long a = (b); a >= 0; --a)
#define f1rd(a, b, c) for (long long a = (b); a >= (c); --a)
#define ms(arr, v) memset(arr, v, sizeof(arr))
#define pb push_back
#define send {ios_base::sync_with_stdio(false);}
#define help {cin.tie(NULL); cout.tie(NULL);}
#define fix(prec) {cout << setprecision(prec) << fixed;}
#define mp make_pair
#define f first
#define s second
#define all(v) v.begin(), v.end()
#define getunique(v) {sort(all(v)); v.erase(unique(all(v)), v.end());}
#define readgraph(list, edges) for (int i = 0; i < edges; i++) {int a, b; cin >> a >> b; a--; b--; list[a].pb(b); list[b].pb(a);}
#define ai(a, n) for (int ele = 0; ele < n; ele++) cin >> a[ele];
#define ain(a, lb, rb) for (int ele = lb; ele <= rb; ele++) cin >> a[ele];
#define ao(a, n) {for (int ele = 0; ele < (n); ele++) { if (ele) cout << " "; cout << a[ele]; } cout << '\n';}
#define aout(a, lb, rb) {for (int ele = (lb); ele <= (rb); ele++) { if (ele > (lb)) cout << " "; cout << a[ele]; } cout << '\n';}
#define vsz(x) ((long long) x.size())
typedef long long ll;
typedef long double lld;
typedef unsigned long long ull;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef vector<ll> vl;
typedef vector<pii> vpi;
typedef vector<pll> vpl;

template<typename A> ostream& operator<<(ostream &cout, vector<A> const &v);
template<typename A, typename B> ostream& operator<<(ostream &cout, pair<A, B> const &p) { return cout << "(" << p.f << ", " << p.s << ")"; }
template<typename A> ostream& operator<<(ostream &cout, vector<A> const &v) {
    cout << "["; for (int i = 0; i < v.size(); i++) {if (i) cout << ", "; cout << v[i];} return cout << "]";
}
template<typename A, typename B> istream& operator>>(istream& cin, pair<A, B> &p) {
    cin >> p.first;
    return cin >> p.second;
}

mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
// mt19937 rng(61378913);
/* usage - just do rng() */

void usaco(string filename) {
    // #pragma message("be careful, freopen may be wrong")
    freopen((filename + ".in").c_str(), "r", stdin);
    freopen((filename + ".out").c_str(), "w", stdout);
}

// const lld pi = 3.14159265358979323846;
// const ll mod = 1000000007;
// const ll mod = 998244353;
// ll mod;



// ll n, m, k, q, l, r, x, y, z;
// const ll template_array_size = 1e6 + 585;
// ll a[template_array_size];
// ll b[template_array_size];
// ll c[template_array_size];
// string s, t;
// ll ans = 0;

// ll findmin(vector <ll> a, ll n) {
//     ll min = 1e10, pos = 0;
//     f0r(i, n)
//     if (a[i] < min) {min = a[i]; pos = i;}
//     return pos;
// }
// vector<bool> v;
// vector<vector<ll>> g;
// map<ll, ll> f, tp;

const int N = 1e5;
// class Node {
//     vector<string> Q;
//     public:
//      int id;
//     public:
//     void AddQuestion(const string& s) {
//         Q.push_back(s);
//     }
//     void printQuestion() {
       
//     }
// };
// vector<Node> nodes(N);
// vector<int> g[N];
// bool vis[N];

vector<string> q[N];
vector<int> g[N];
vector<vector<int>> response;
ll cnt = 0;
/*
Array locations with particular numbers gives rise to output for that node.
Example - ABCDEFG
          0xxx11x => 1 for that node (ABC.. are questions)
*/
void dfs(int a,int p) {
    vector<int> temp;
   for(string i:q[a]){
    cout<<i<<"\n";
    int t;
    cin>>t;
    temp.push_back(t);
   }
   response.push_back(temp);
   temp.clear();
   for(int i:g[a])if(i!=p)dfs(i,a);
}
/*
Further scope: 
Read questions dynamically from a pool of questions uploaded in a file (editable and viewable).
Implement questions as graphs as well.
*/
int main() {
    string ans, out[10];
    cout<<"For diagnosing depression, we use SCID-5-CV as default. Do you wish to change the rules? (Y/N)";
    cin>>ans;
    if(ans=="n")
    {
        int result = system("scid_baseline.exe");
        return 0;
    }
    int n_out;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    {
        cout<<"Enter the number of output categories:";
        cin>>n_out;
        f0r(i, n_out)
        {
            cout<<"Enter diagnosis for output class "<<i+1<<" :";
            cin>>out[i];
        }
    }
    else 
    {
        n_out = 4;
        out[0] = "Not diagnosed with Major Depressive Disorder!!!";
        out[1] = "Diagnose: Depressive Disorder Due to AMC!!!";
        out[2] = "Diagnose: Substance-Induced Depressive Disorder!!!";
        out[3] = "Diagnose: CURRENT MAJOR DEPRESSIVE EPISODE!!!";
    }
    int n, m;
    cout<<"Enter the number of nodes and edges:";
    cin >> n >> m;
    cout<<"Enter the node numbers (starting from 1) which are connected in groups of two separated by a new line. E.g.- 1 2\n2 3\n";
    for(int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        u--,v--;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    // for(int i = 0; i < n; ++i) {
    //     for(int j:g[i])
    //         cout<<j<<" ";
    //     cout<<"\n";
    // }
    for(int i=0;i<n;i++){
        int c=0;
        cout<<"Enter number of questions for "<<i+1<<"th node:\n";
        cin>>c;
        cout<<"Please enter the questions"<<endl;
        for(int j =0;j<c;j++){
            string s;
            cin>>s;
            q[i].push_back(s);
        }
           
    }
    vector<string> rules(n);
    cout<<"\nNow for each node (newline separated) input the question number along with the corresponding response that increments count for that node or add an extra index to point to the output category, as needed.\nE.g.- 1121312\n105161\n";
    for(int i=0;i<n;i++)
        cin>>rules[i];
    cout<<"Please answer the following questions with either 0 or 1.\n";
    dfs(0,-1);
    // for(auto t: response)
    //     cout<<"\n"<<t;

    /*
    Right now only a unique combination of responses lead to count increment. To be changed later.
    */
    for(int k = 0; k<n; k++)
    {
        int flag = 0;
        int j = rules[k].size();
        for(int t = 0; t<j-1; t=t+2)
        {
            if(response[k][(rules[k][t]-'0')-1]!=(rules[k][t+1])-'0')
            {
                flag = 1;
                break;
            }
            
        }
        
        if(j%2!=0)
        {
            if(flag==0)
            {   
                cout<<out[(rules[k][j-1]-'0')];
                return 0; 
            }
            else
                cnt++;
        }
        
        if(flag == 0)
            cnt++;
    }

    /*
    Note: By default count greater than 5 leads to last output label.
    */
    if(cnt>5)
        cout<<out[n_out-1];
    return 0;

}
