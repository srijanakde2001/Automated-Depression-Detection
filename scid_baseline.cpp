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

const lld pi = 3.14159265358979323846;
const ll mod = 1000000007;
// const ll mod = 998244353;
// ll mod;



ll n, m, k, q, l, r, x, y, z;
const ll template_array_size = 1e6 + 585;
ll a[template_array_size];
ll b[template_array_size];
ll c[template_array_size];
string s, t;
ll ans = 0;

ll findmin(vector <ll> a, ll n) {
    ll min = 1e10, pos = 0;
    f0r(i, n)
    if (a[i] < min) {min = a[i]; pos = i;}
    return pos;
}
vector<bool> v;
vector<vector<ll>> g;
map<ll, ll> f, tp;
void bfs(ll u, ll tm)
{
    queue<ll> q;
    q.push(u);
    // v[u] = true;

    while (!q.empty()) {
        if (tm <= 0)return;
        ll f = q.front();
        q.pop();
        for (auto i = g[f].begin(); i != g[f].end(); i++) {
            q.push(*i);
            tp[*i] = 1;
        }
        tm--;
    }
}

int main() {

int count = 0, flag = 0;
string ans;

    // <-------------------A1------------------->

cout<<"\nIn the past month, since (ONE MONTH AGO), has there been a period of time when you were feeling depressed or down most of the day, nearly every day? (Has anyone said that you look sad, down, or depressed?)\n";
cin>>ans;
if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
{
    cout<<"\nHow about feeling sad, empty, or hopeless, most of the day, nearly every day?";
    cin>>ans;
}
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nWhat has it been like? How long has it lasted? (As long as 2 weeks?)";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A2------------------->

if(count==1)
{
    cout<<"\nDuring that time, did you have less interest or pleasure in things you usually enjoyed? (What has that been like?)";
    cin>>ans;
}
else if(count==0)
{
    cout<<"\nWhat about a time since (ONE MONTH AGO) when you lost interest or pleasure in things you usually enjoyed? (What has that been like?)";
    cin>>ans;
}
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nHas it been nearly every day? How long has it lasted? (As long as 2 weeks?)";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}
if(count==0)
{
    cout<<"\nNot diagnosed with Major Depressive Disorder!!!";
    return 0;
}

    // <-------------------A3------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) how has your appetite been? (What about compared to your usual appetite? Have you had to force yourself to eat? Eat [less/more] than usual? Has that been nearly every day? Have you lost or gained any weight?)";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nHow much? (Had you been trying to [lose/gain] weight?)";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A4------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) how have you been sleeping? (Trouble falling asleep, waking frequently, trouble staying asleep, waking too early, OR sleeping too much?)";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nHow many hours of sleep (including naps) have you been getting? How many hours of sleep did you typically get before you got (depressed/OWN WORDS)? Has it been nearly every night?";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A5------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) have you been so fidgety or restless that you were unable to sit still?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    flag = 1;
cout<<"\nWhat about the opposite - talking more slowly, or moving more slowly than is normal for you, as if you're moving through molasses or mud?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    flag = 1;
if(flag==1)
{
    flag = 0;
    cout<<"\nIn either instance, has it been so bad that other people have noticed it? What have they noticed? Has that been nearly every day?";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A6------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) what was your energy like? (Tired all the time? Nearly every day?)";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    count += 1;

    // <-------------------A7------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) have you been feeling worthless?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    flag = 1;
cout<<"\nWhat about feeling guilty about things you have done or not done?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nWhat kinds of things? (Is this only because you can't take care of things since you have been sick?)";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        flag = 1;
}
if(flag==1)
{
    flag = 0;
    cout<<"\nNearly every day?";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A8------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) have you had trouble thinking or concentrating? Has it been hard to make decisions about everyday things? (What kinds of things has it been interfering with? Nearly every day?)";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    count += 1;

    // <-------------------A9------------------->

cout<<"\nDuring (THE WORST 2-WEEK PERIOD OF THE PAST MONTH) have things been so bad that you thought a lot about death or that you would be better off dead? Have you thought about taking your own life?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nHave you done something about it? (What have you done? Have you made a specific plan? Have you taken any action to prepare for it? Have you actually made a suicide attempt?)";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        count += 1;
}

    // <-------------------A10------------------->

if(count<5)
{
    cout<<"\nNot diagnosed with Major Depressive Disorder!!!";
    return 0;
}

    // <-------------------A11------------------->

cout<<"\nHow have (DEPRESSIVE SXS) affected your relationship or your interactions with other people? (Have [DEPRESSIVE SXS] caused you any problems in your relationships with your family, romantic partner, or friends?)";
cin>>ans;
if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
{
    cout<<"\nHow have (DEPRESSIVE SXS) affected your work/school? (How about your attendance at work/school? Have [DEPRESSIVE SXS] made it more difficult to do your work/schoolwork? Have [DEPRESSIVE SXS] affected the quality of your work/schoolwork?)";
    cin>>ans;
    if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
    {
        cout<<"\nHow have (DEPRESSIVE SXS) affected your ability to take care of things at home? How about doing simple everyday things, like getting dressed, bathing, or brushing your teeth? What about doing other things that are important to you, like religious activities, physical exercise, or hobbies? Have you avoided doing anything because you felt like you weren't up to it?";
        cin>>ans;
        if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
        {
            cout<<"\nHave (DEPRESSED SXS) affected any other important part of your life?";
            cin>>ans;
            if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
            {
                cout<<"\nHow much have you been bothered or upset by having (DEPRESSIVE SXS)? A lot?";
                cin>>ans;
                if(ans=="No" || ans=="NO" || ans=="N" || ans=="n")
                {
                    cout<<"\nNot diagnosed with Major Depressive Disorder!!!";
                    return 0;
                }
            }
        }

    }
    
}

    // <-------------------A12------------------->

cout<<"\nJust before this period of depression began, were you physically ill?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nWere you suffering from one or more of the following conditions: stroke, Huntington's disease, Parkinson's disease, traumatic brain injury, Cushing's disease, hypothyroidism, multiple sclerosis, systemic lupus erythematosus?";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    {
        cout<<"\n<-------------------   Diagnose: Depressive Disorder Due to AMC!!!   ------------------->";
        return 0;
    }
}
cout<<"\nJust before this began, were you taking any medications or drinking or using any street drugs?";
cin>>ans;
if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
{
    cout<<"\nWere you taking any one or more of the following: alcohol(I/W); phencyclidine (I); hallucinogens (I); inhalants (I); opiods (I/W); sedatives, hypnotics, or anxiolytics (I/W); amphetamine and other stimulants (I/W); cocaine (I/W); antiviral agents (efavirenz); cardiovascular agents (clonidine, guanethidine, methyldopa, reserpine); antidepressants; anticonvulsants; antimigrane agents (triptans); antipsychotics; hormonal agents (corticosteroids, oral contraceptives, gonadotropin-releasing hormone agonists, tamoxifen); smoking cessation agents (varenicline); immunological agents (interferon) ?";
    cin>>ans;
    if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
    {
        cout<<"\nAny change in the amount you were taking?";
        cin>>ans;
        if(ans=="Yes" || ans=="YES" || ans=="Y" || ans=="y")
        {
            cout<<"\n<-------------------   Diagnose: Substance-Induced Depressive Disorder!!!   ------------------->";
            return 0;
        }
    }
}
cout<<"\n<-------------------   Diagnose: CURRENT MAJOR DEPRESSIVE EPISODE!!!   ------------------->";
return 0;

}