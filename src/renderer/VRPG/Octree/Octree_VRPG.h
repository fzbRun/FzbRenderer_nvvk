#pragma once

#include "feature/Feature.h"
#include "./OctreeShaderio_VRPG.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_OCTREE_2_VRPG_H
#define FZBRENDERER_OCTREE_2_VRPG_H
namespace FzbRenderer {
struct OctreeCreateInfo_VRPG {
	std::vector<nvvk::Buffer> VGBs;       // geometry grid
	shaderio::float3 VGBStartPos;
	shaderio::float3 VGBVoxelSize;
	float VGBSize;

	std::vector<nvvk::Buffer> lightVGBs;  // light grid, independent resolution
	shaderio::float3 lightVGBStartPos;
	shaderio::float3 lightVGBVoxelSize;
	float lightVGBSize;

	AccelerationStructureManager* asManager;
};

class Octree_VRPG : public Feature {
public:
	Octree_VRPG() = default;
	virtual ~Octree_VRPG() = default;

	Octree_VRPG(pugi::xml_node& featureNode);

	void init(OctreeCreateInfo_VRPG createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void renderStatic(VkCommandBuffer cmd);
	void renderCandidates(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createOctreeArray();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initOctreeArray(VkCommandBuffer cmd);
	void createOctreeArray(VkCommandBuffer cmd);
	void createLightTree(VkCommandBuffer cmd);
	void getOctreeLabel(VkCommandBuffer cmd);
	void getOctreeNodePairData(VkCommandBuffer cmd);
	void getNearbyClusterInfo(VkCommandBuffer cmd);

	shaderio::OctreePushConstant_VRPG pushConstant{};

	uint32_t octreeMaxLayer = 6;
	uint32_t lightOctreeMaxLayer = 4;

	float clusterFillRateThreshold = 0.75f;


	std::vector<nvvk::Buffer> octreeNodeDataBuffer_G;
	std::vector<nvvk::Buffer> octreeNodeInfoBuffer_G;			//label_indivisible,  label: indivisible node闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨绘い鎺嬪灪閵囧嫰骞囬姣挎捇鏌熸笟鍨妞ゎ偅绮撳畷鍗炍旈埀顒勫煕閹烘鈷戠紓浣姑粭鍌滅磼椤旂晫鎳囩€殿喖顭锋俊鍫曞幢濡搫浼庨梻浣哥秺閸嬪﹪宕楀鈧畷鎴﹀箻鐠囨彃鍞ㄥ銈嗗姉閸犲孩绂嶉悙顒佸弿婵妫楁晶濠氭煟閹剧偨鍋㈤柡灞糕偓宕囨殕閻庯綆鍓涜ⅵ闁诲孩顔栭崳顕€宕抽敐鍛殾闁圭儤鍩堝鈺傘亜閹达絾纭堕悽顖涚〒缁辨捇宕掑顑藉亾妞嬪孩濯奸柡灞诲劚绾惧鏌熼悙顒€澧柣鏂挎閹綊宕崟顒佸創濡炪倧瀵岄崜鐔煎蓟濞戞埃鍋撻敐搴′簼鐎规洖鐬奸埀顒冾潐濞叉﹢銆冮崱妤婂殫闁告洦鍓涚弧鈧繛杈剧到婢瑰﹤螞濠婂牊鈷掗柛灞捐壘閳ь剟顥撳▎銏狀潩鐠鸿櫣鍔﹀銈嗗笒閸婂憡绂掑鍫熺厾婵炶尪顕ч悘锟犳煛閸涱厾鍩ｉ柟顔ㄥ洤閱囨繝濠傛噽閻ｉ箖姊绘担鍛婂暈闁荤喆鍎佃棟妞ゆ牗绋掗崣蹇曗偓骞垮劚椤︿即鎮″▎鎰╀簻闁哄啫鍊瑰▍鏇㈡煙閸愬弶鍤囬柡灞剧洴閹晝鎷犺娴兼劙鏌ф导娆戠М鐎殿喖鐖奸幃娆戔偓娑欘焽閸戔€愁渻閵堝啫鍔滅€光偓缁嬫娼栫紓浣股戞刊鎾煟閻旂厧浜伴柛銈咁儑缁辨挻鎷呯粵瀣闂佺锕ょ紞濠囧Υ閸愵喖唯闁宠桨鑳堕鎰箾鏉堝墽鎮兼い顓炵墦瀹曠敻骞嬪┑鍐╂杸闂佺粯鍔栧娆撴倶閿旇姤鍙忔俊顖滎焾婵″灝顭跨憴鍕缂佽桨绮欏畷銊︾節閸曨偄绗氶梺鑽ゅ枑缁秶鍒掗幘宕囨殾婵犲﹤鐗婇崑鍕渻鐎ｎ亜顒㈡繛鍫㈠亾缁绘繈妫冨☉妯峰亾婵犳埃鈧箓宕奸妷锕€浜楅梺纭呮彧闂勫嫰鎮￠崘顔肩骇闁绘劙娼ч。鍏笺亜閿曗偓閸㈡煡鈥︾捄銊﹀枂闁告洦鍓涢ˇ銉╂煣缂佹澧甸柡宀嬬到椤粓鍩€椤掆偓椤洩顦崇紒鍌涘笒椤劑宕奸悢鍝勫箺闂傚鍋勫ú锕€顫忛懡銈囦笉闁绘劗顣介崑鎾舵喆閸曨剛顦ㄧ紓浣筋嚙閸婂潡鏁愰悙娴嬫斀閻庯綆鍋勫▓婵嬫⒑濮瑰洤鐏叉繛浣冲洨宓佺€广儱顦伴埛鎺懨归敐鍥ㄥ殌闁告柣鍊栫换娑氣偓娑櫭崫铏光偓瑙勬礃閸ㄥ灝鐣烽悢纰辨晢濞达絿顭堟导搴ㄦ⒒娴ｈ鍋犻柛搴㈡そ瀹曟粌顫濋懜闈涗簵闂侀潧顦弲婊堟偂濞戙垺鐓曢柕澶樺枤閸樻垶銇勮箛鎾跺闁稿﹤鐖奸弻娑樷枎韫囷絾鈻撳┑鐐茬墔缁瑩寮婚敐澶婄疀妞ゆ牗绋戝☉褔姊洪懖鈺侇暭閻庣瑳鍥﹂柛鏇ㄥ灠椤懘鏌ㄥ☉妯侯仾闁革絼鍗冲铏规嫚閳ヨ櫕鐏堟繝鈷€鍡椥撶紒鍌涘浮閺佸啴宕掑☉妯兼濠电姰鍨煎▔娑㈡儗閸儱鑸归柛顐ｆ礃閳锋帡鏌涚仦鎹愬闁逞屽墯閹倸鐣烽幇顓фЧ閹艰揪绲块悞鍏肩箾閹炬潙鐒归柛瀣尰椤ㄣ儵鎮欏顔解枅闂佹悶鍔屾晶浠嬪箹閻愵剚濯撮悷娆忓娴犮垽姊虹拠鈥虫灁闁搞劏娅ｇ紓鎾绘嚍閵夈儳绐為柣蹇曞仩濡嫭顨欓梻鍌欐祰瀹曠敻宕伴幇顔煎灊鐎广儱鎲樺☉銏犵妞ゆ梹鍎崇粊锕傛⒑閸涘﹤濮﹂柣鎾崇墕椤洭濡搁埡鍌滃帗閻熸粍绮撳畷婊堟偄婵傚娈ㄥ銈嗗姧缁犳垹绮诲☉銏＄厸濠㈣泛锕︽禒銏ゆ煠閺夋寧鍋ユ慨濠囩細閵囨劙骞掗幋婊冩瀳濠电偛鐡ㄧ划搴ㄥ磻閵堝绠氶柛鏇ㄥ墯閸庣喖鏌曟繝蹇氬悅闁归绮换娑欐綇閸撗勫仹闂佺绨洪崐婵嗙暦閻㈢鐒垫い鎺戝閳锋垿鎮楅惂鎼佹婵犮垺蓱娣囧﹪宕稿Δ浣哄幍濡ょ姷鍋涢悘婵嬫倶閼碱兘鍋撶憴鍕闁挎洏鍨介妴浣糕枎閹炬潙娈愰梺鍐叉惈閸燁偊鎮欐繝鍌楁斀闁绘ɑ顔栭弳顖涗繆閹绘帗鍤囩€规洝顫夌€佃偐鈧稒锚閸擃參姊虹紒妯虹伇婵☆偄瀚竟鏇熺節濮橆厾鍘甸梺纭咁潐閸旀洟鎮￠幇鐗堢厱闁绘ê寮堕ˉ銏ゆ煛鐏炲墽娲寸€殿喗鎸虫俊鎼佸Ψ閵夘喗楠勫┑锛勫亼閸娧呮嫻閻旂厧绀夐柟瀛樼箥濞兼牗绻涘顔荤凹闁搞倖甯￠弻鏇㈠醇濠靛洤娅ゅ┑顕嗙稻閸旀瑥顫忓ú顏勪紶闁告洦鍓欏▍褎绻濋悽闈涗汗闁稿鎹囧铏圭磼濡湱绻侀梺闈╃秶缁蹭粙鎮惧畡閭︽建闁逞屽墴閵嗕礁鈻庨幘鏉戞異闂佸疇妗ㄧ粈浣虹不娴煎瓨鈷掗柛灞剧懅椤︼箓鏌熷ù瀣⒉缂佹鍠庤灃闁告侗鍘鹃悰銉モ攽椤旂瓔鐒炬繛澶嬬〒缁牊寰勯幇顓犲幈濡炪倖鍔х徊鍓х矆閸愵喗鐓曞┑鐘插€绘禒娑欍亜閵婏絽鍔﹂柟顔界懇閹崇娀顢栭懗顖涱€楅梻鍌欑閹测€愁潖閻熸壆鏆嗛柟闂寸閽冪喖鏌ｉ弮鍌氬付缂佲偓閸愵喗鐓犵痪鏉垮船婢х増銇勯弮鈧敮鈥愁潖閾忚瀚氶柍銉ㄦ珪閻忓秹鏌ｆ惔锝囨嚄闁告侗鍨抽悡鏃堟⒒娓氬洤澧紒澶屾暬瀹曟洖顓兼径瀣帗閻熸粌绻橀獮鍡涘礆閻熷崨sibleNode闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱濠电姴鍊归崑銉╂煛鐏炶濮傜€殿噮鍣ｅ畷濂告偄閸涘鍞堕梻鍌欒兌椤牓顢栭崱娑樼闁告挆鍐ㄧ亰濡炪倖鎸鹃崑鎰ｉ崼鐔剁箚妞ゆ牗绻嶉崵娆愮箾閸涘洤娲﹂埛鎴炵箾閼奸鍤欐鐐搭殜閺岋綁鎮㈤崣澶嬬彋閻庢鍠栭…鐑藉箖閵忋垺鍋橀柍銉ュ帠婢规洟姊哄Ч鍥х仾妞ゆ梹鐗犻幃鐐哄箚椤€崇秺閹晛鈻庤箛鏂款槱闂佹娊鏀辩敮锟犲蓟濞戙垹鍗抽柕濞垮劙缁ㄨ顪冮妶鍛闁告瑥鍟村濠氬灳瀹曞洦娈曢柣搴秵閸撴盯鎯侀崼銉﹀€甸悷娆忓缁€鍐煥閺囨ê鐏ǎ鍥э躬楠炴牗鎷呴懖婵勫姂閺屾洝绠涢弴鐐愨€愁熆鐟欏嫭绀嬫慨濠囩細閵囨劙骞掗幙鍕惞缂傚倷璁查崑鎾炽€掑锝呬壕閻?闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊炲銈嗗笒椤︿即寮查鍫熷仭婵犲﹤鍟扮粻濠氭煕閳规儳浜炬俊鐐€栫敮濠囨嚄閸洖鐓濋柟鍓х帛閻撴盯鏌涘☉鍗炵仩闁宠鐗撻弻鏇㈠幢閺囩媭妲梺瀹狀嚙闁帮綁鐛鈧畷姗€骞撻幒鎾存婵犵绱曢崑鎴﹀磹閺嶎厼绠板Δ锝呭暙绾惧鏌ｉ弬鍨倯闁稿锕㈤弻鏇熷緞閸℃ɑ鐝曢梺鎶芥敱閸ㄥ潡寮诲☉銏℃櫆閻犲洦褰冪粻褰掓⒑閸涘﹥鐓ユい鎴濇婵＄敻宕熼姣尖晠鏌曟径娑氱暠婵炲憞鍕瘈闁靛骏缍嗗鎰箾閸欏鐒介柟骞垮灩閳规垿宕遍埡鍌氬厞婵＄偑鍊栭幐鐐叏閹绢喚宓佺€广儱顦伴埛鎺懨归敐鍫燁仩閻㈩垱鐩弻娑㈠籍閹惧墎鏆ら悗瑙勬礃缁矂锝炲┑瀣殝闂傚牊绋堥崑鎾诲醇閺囩啿鎷哄銈嗗坊閸嬫挾绱掓径灞炬毈闁诡喒鈧枼妲堥柕蹇ョ磿閸樻捇鎮峰鍕煉鐎规洘绮撻幃銏ゆ偂鎼达絿鏋冮梻浣虹帛閸旀洟骞栭銏℃殰闂傚倷鐒︾€笛呮崲閸屾壕鍋撳鐓庢珝闁诡垰瀚幆鏃堝Ω閿旇瀚藉┑鐐舵彧缁蹭粙骞夐敓鐘茬畾闁割偁鍎查悡鏇㈡煟閺囨氨顦﹀ù婊勫閳ь剝顫夊ú姗€宕濆▎鎾崇畺闁伙絽鐬奸惌娆撴偣娓氼垳鍘涢柛鐐垫暬濮婄粯鎷呴懞銉с€婇梺鍝ュУ閹稿骞堥妸鈺傚仺闁告稑锕ゆ禍妤呮⒑閸濆嫭鍌ㄩ柛銊ユ贡缁崵鎲撮崟顓狀啎闂佹寧绻傛鍛婃櫠椤曗偓閺岋綀绠涢敐鍛彎闂佸搫鐭夌紞渚€銆佸鈧幃娆撴偨閸忓摜绀冪紓鍌氬€峰ù鍥ㄣ仈閸濄儲鏆滄俊銈傚亾妞ゎ厼娲╃粻娑樷槈濡壕鏅犻弻鏇熺珶椤栨碍鍣洪柛銈呭暙閳规垿鎮╅崹顐ｆ瘎闂佺顑囬崰鏍х暦閵忥綆妲鹃梺鐟扮畭閸ㄥ綊锝炲鍫濈劦妞ゆ巻鍋撻柣锝呭槻鐓ゆい蹇撴噺濞呫垽姊虹紒姗嗙劸閻忓繑鐟╅弫宥咁煥閸愶絾鏂€闂佺粯顭堥婊冾啅閵夆晜鐓欓柛鎰皺缁犵増銇勯鐐村仴鐎殿喛娉涢～蹇涘绩婵℃sible闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱濠电姴鍊归崑銉╂煛鐏炶濮傜€殿噮鍣ｅ畷濂告偄閸涘鍞堕梻鍌欒兌椤牓顢栭崱娑樼闁告挆鍐ㄧ亰濡炪倖鎸鹃崑鎰ｉ崼鐔剁箚妞ゆ牗绻嶉崵娆愮箾閸涘洤娲﹂埛鎴炵箾閼奸鍤欐鐐搭殜閺岋綁鎮㈤崣澶嬬彋閻庢鍠栭…鐑藉箖閵忋倕绀傞柣鎾崇岸閸嬫挾绱掑Ο璇插伎濠德板€愰崑鎾翠繆椤愶絿鎳囬柟顕€绠栭幃婊堟嚍閵夈垺瀚奸梻浣侯攰閸嬫劙宕戝☉銏犵劦妞ゆ帊鐒︾粈瀣煙瀹曞洤鏋涙い銏＄☉閳规垿宕煎鍛板厭闂傚倸鍊搁崐鐑芥倿閿旈敮鍋撶粭娑樻噺瀹曟煡鏌涘畝鈧崑娑氱不閻樿绠规繛锝庡墮婵″吋鎱ㄩ悷鏉跨骇缂佺粯鐩畷鍗炍熺拠鏌ョ€洪梻浣虹帛閹歌煤閻旂厧钃熼柡鍥╁枔缁犻箖鏌涢…鎴濇灈濠殿喖閰ｅ娲焻閻愯尪瀚板褜鍨堕弻鐔碱敊閻撳簶鍋撻幖浣规櫇闁靛骏绱曢悷褰掓煃瑜滈崜娆擄綖韫囨稒鎯為柛锔诲幘閿涙粌鈹戦埥鍡楃仩婵炲瓨纰嶇粋宥咁煥閸℃瑤鑸繝鐢靛Х閺佸憡鎱ㄩ弶鎳ㄦ椽濡堕崶褜妫滈梺绉嗗嫷娈橀柛娆愭崌閺屾盯濡烽敐鍛瀴闂佺粯鎼徊璺ㄦ閹烘梹瀚氶柛娆忣樈濡繝姊洪棃娑欐悙閻庢矮鍗抽悰顕€宕堕鈧粈鍫澝归敐鍥ㄧ＊闁靛牆顦伴埛鎴︽煕韫囨挸鎮戠紒璺哄级缁绘稓娑垫搴ｇ槇閻庢鍠栭…鐑藉垂閹呮殾闁搞儯鍔嶉悾鐑芥⒑鐠囨彃鍤辩紓宥呮瀹曟粌鈽夊┃鎯у緧闂傚倸鍊搁崐宄懊归崶顒夋晪闁哄稁鍘肩粣妤呮煙閻戞﹩娈旈柣銈囧亾缁绘盯骞嬪▎蹇曚患闂佺粯鎸荤换鍫ュ蓟閻旂厧绠查柟浼存涧濞堫厾绱撴担鍝勑￠柛妤佸▕閻涱噣寮介銏犵亰闂佺绻愰ˇ顖涚妤ｅ啯鐓涢柛灞久崝姘辨喐閹殿喖姣榚闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨奸柟鐧哥秮閺岋綁顢橀悙鎼闂侀潧妫欑敮鎺楋綖濠靛鏅查柛娑卞墮椤ユ艾鈹戞幊閸婃鎱ㄩ悜钘夌；闁绘劗鍎ら崑瀣煟濡崵婀介柍褜鍏涚欢姘嚕閹绢喖顫呴柣妯荤垹閸ャ劎鍘遍柣蹇曞仜婢т粙鎮￠姘肩唵閻熸瑥瀚粈鈧梺瀹狀潐閸ㄥ潡銆佸▎鎾村剹妞ゆ劦鍋傜花濠氭煟閿濆洤鍘存い銏☆殜瀹曠喖顢曢姀鐘辨喚闂備浇顕уù鐑藉极閸濄儲鍏滈柛顐ｆ礀绾惧鏌熼幑鎰厫闁哥姴妫濋弻娑㈠即閵娿儱顫銈忚礋閸庡磭妲愰幘璇茬＜婵ɑ鐦烽姀銈嗙叆闁哄洢鍔嬮柇顖氣攽閳ュ磭鎽犻柟宄版噽閸栨牠寮撮悤浣圭秾濠电姵顔栭崰妤呫€冮崨顓囨稑鈽夐姀鐘电厬闂佺硶鍓濈粙鎺楀煕閹达附鐓曟繛鎴烇公濮婃绱掗崼鏇氭喚闁哄本鐩俊鎼佹晝閳ь剟鍩㈤弴鐘亾濞堝灝鏋熷┑鐐诧攻娣囧﹪鎮滈挊澹┿劑鏌曡箛濠傚⒉闁哄鎮傚缁樻媴閾忕懓绗￠梺鎸庢磵閺呯姴鐣锋导鏉戝唨妞ゆ挾鍠庨埀顒€鐖煎娲敆閳ь剛绮旈悽鍛婂亗闁哄洢鍨洪悡鍐煕濠靛棗顏╅柍褜鍓涚划顖炲疾閸洘鍋ㄧ痪鐗埫禍楣冩偡濞嗗繐顏紒鈧崘顏嗙＜閻犲洩灏欐晶鏇熴亜閺囶亞绉鐐达耿椤㈡瑧鍠婃潏銊хП闂傚倷鑳剁划顖炩€﹂崶顒€鏋侀柟闂寸閸屻劍銇勯幒鍡椾壕闂佸疇顫夐崹鍧楀箖閳哄懎鍨傛い鎰剁稻閻﹀骸鈹戦悩鎰佸晱闁哥姵顨呴…鍥р枎閹存繂鐤鹃梻鍌欑劍鐎笛兠洪弽顓炵９鐟滅増甯掔壕濠氭煕瀹€鈧崑鐐烘偂閻斿吋鐓涢柛灞炬皑娴犮垽鏌熼钘夌伌闁哄矉缍侀獮妯虹暦閸モ晩鍞舵俊鐐€ら崣鈧繛澶嬫礋楠炲繘鎮╃拠鑼紜閻庤娲栧ú锝夊传濡ゅ懏鈷掑ù锝呮啞閹叉悂鏌涢悩铏鞍闁哄懓娉涜灃闁告劦浜為悞鍧楁倵楠炲灝鍔氭い锔诲灣閻ヮ亣顦查柍钘夘樀瀵噣宕掑☉娆戝建isible闂?
	std::vector<nvvk::Buffer> octreeNodeDataBuffer_E;

	nvvk::Buffer globalInfoBuffer;
	nvvk::Buffer indivisibleNodeInfosBuffer_G;					//闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨奸柟鐧哥秮閺岋綁顢橀悙鎼闂傚洤顦甸弻銊モ攽閸♀晜效婵炲瓨鍤庨崐婵嬪蓟閵堝绾ч柟绋块娴犳潙鈹戦纭锋敾婵＄偘绮欓妴浣肝旈崨顓犲姦濡炪倖甯掗崐濠氭儗閹剧粯鐓涢悘鐐额嚙閸旀岸鏌ｉ幒鎴犱粵闁靛洤瀚伴獮鎺楀幢濡炴儳顥氶梻鍌欒兌椤牓鏁冮妶鍜冭€块弶鍫氭櫅閸ㄦ繈鎮楅敐搴℃灈缂佺姵濞婇弻锟犲磼濮橆厽鍎撴繛瀛樼矒缁犳牠寮婚妸銉㈡斀闁糕剝鐟ラ崵顒傜磽娴ｉ潧濡虹紒顔界懇瀵濡搁妷銏☆潔濠电偛妫欓崹鍫曞礆濞戞ǚ鏀芥い鏃傘€嬮弨缁樹繆閻愭壆鐭欐鐐插暣閹粓鎸婃径宀勭崜闂佸湱鍘ч悺銊╁箰閼姐倐鍋撻崹顐ゅ⒈缂佽鲸鎹囧畷鎺戔枎閹达絿鐛ラ梻渚€娼荤紞鍡涘闯閿濆懐鏆︾憸鐗堝笒闁卞洭鏌￠崶鈺佷户闁伙絾妞藉娲倷閽樺濮曢梺鍝勵儑閸樠冨祫闂佸壊鍋侀崕鏌ユ偂韫囨稓鍙撻柛銉ｅ劚閸斻倗鐥幆褎鍋ラ柡灞剧☉铻ｉ柛婵嗗閸╃偛顪冮妶搴濈盎闁哥喎鐡ㄦ穱濠囧醇閺囩偛鑰垮┑掳鍊愰崑鎾舵偘閼测晝纾介柛灞剧懅椤︼妇鈧鍠栨晶搴ㄥ焵椤掑倻鎳楅柛鏇ㄥ幐閺€铏節閻㈤潧孝婵炲眰鍊濋幃娆愮節閸愶缚绨婚梺鐟版惈缁夌兘宕楀畝鍕厱婵☆垳绮畷宀勬煙椤旂厧妲绘い顓滃姂瀹曠喖顢橀悩闈涘辅闂佽姘﹂～澶娒哄Ο鐓庡灊鐎光偓閸曞墎绋忓┑鐘绘涧椤戞劙寮崟顖涚厓闁告繂瀚埀顒€顭烽崺銉﹀緞鐎ｃ劋绨诲銈嗗姧缂嶅棗鈻撻弮鈧妵鍕敆閳ь剟鏁冮敃鍌氱劦妞ゆ巻鍋撻柛姘ｅ亾閻庡綊娼荤粻绌抌leNode闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨奸柟鐧哥秮閺岋綁顢橀悙鎼闂傚洤顦甸弻銊モ攽閸♀晜效婵炲瓨鍤庨崐婵嬪蓟閵堝绾ч柟绋块娴犳潙鈹戦纭锋敾婵＄偘绮欓妴浣肝旈崨顓犲姦濡炪倖甯掗崐濠氭儗閹炬番浜滈柡鍥殔娴滈箖姊虹化鏇熸澒闁稿鎸搁—鍐Χ閸℃鐟ㄩ梺鎸庢穿缁犳捇骞冩ィ鍐╁仺缁剧増锚娴滈箖鎮峰▎蹇擃仾缂佲偓閳ь剟姊洪棃娑氬闁规祴鍓濈粚杈ㄧ節閸愮偓效闁圭厧鐡ㄩ敋濞存粓绠栭弻銊モ攽閸℃瑥鍤紓浣靛妼閵堟悂寮诲☉銏″亹缂佹稓顢婇埀顒€娼￠弻鈥崇暆鐎ｎ剛锛熸繛瀵稿缁犳捇骞婇悙鍝勎ㄦい鏃傜摂濡冣攽閿涘嫬浜奸柛濠冪墵楠炴劙骞栨担鑲澭呯磼鐎ｎ厽纭堕柡鍡畵閺屾洘绻涢悙顒佺彟闂佽桨绀佺粔鐢垫崲濠靛洨绡€闁稿本纰嶅▓顓犵磽娴ｅ搫校闁搞劌鐏濋～蹇曠磼濡顎撶紓浣圭☉椤戝懎鈻撻鐐╂斀闁绘劕妯婇崵鐔兼煕韫囨棑鑰块柣娑卞枟缁绘繈宕惰椤旀帡鏌ｉ悩鍙夌┛閻忓繑鐟╅幃鍧楀川鐎涙ǚ鎷洪梺鍝勫€堕崕鎻掆枍閸涘瓨鐓曢柣鏃囨硾瀹撳棝鏌曢崱妤佸殗鐎殿喖顭锋俊鐑芥晜閹冪闂備胶顢婇崑鎰嫚濠靛绾у璺烘啞椤樄闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鎯у⒔閹虫捇鈥旈崘顏佸亾閿濆簼绨奸柟鐧哥秮閺岋綁顢橀悙鎼闂傚洤顦甸弻銊モ攽閸♀晜效婵炲瓨鍤庨崐婵嬪蓟閵堝绾ч柟绋块娴犳潙鈹戦纭锋敾婵＄偘绮欓妴浣肝旈崨顓犲姦濡炪倖甯掗崐濠氭儗閹剧粯鍊甸柣銏☆問閻掔偓銇勯妷锕€鍝洪柡灞剧椤﹁櫕銇勯妸銉︻棦闁诡啓銈囩杸婵炴垶鐟ч崢鍗烆渻閵堝棗濮х紒鑼舵硶缁寮借閻斿棝鎮归崫鍕儓妞ゅ浚鍙冮弻鈥崇暆鐎ｎ剛鐦堥悗瑙勬礀瀹曨剝鐏掗梺鍏肩ゴ閺呮繈宕㈠鍫熲拻濞达絽鎲￠幆鍫ユ煟椤掆偓閵堢鐣锋导鏉戝唨妞ゆ挾鍋熼悾鍝勨攽閻樿宸ラ柛姘耿閹垻鎷嬮懗顖涱仩缂佽鲸甯掕灒闂佸灝顑呴惁鑸电節閻㈤潧啸闁轰礁鎲￠幈銊╂偨閸撳弶鏅╅梺鍝勭▉閸樿偐绮ｅΔ浣虹闁瑰鍊戝璺哄嚑鐎广儱顦伴悡娑㈡煕閵夛絽鍔氶柣蹇婃櫅椤╁ジ宕ㄧ€涙ǚ鎷洪梺鍛婄☉閿曘儲寰勯崟顖涚厱闁靛绠戦ˉ瀣磼椤斿墽甯涚紒缁樼箓椤繈顢橀悩鎻掔疄濠电姴鐥夐弶搴撳亾閺囥垹鐤い鎰剁悼椤╅鎲稿澶嬬畳闂備胶绮…鍫濈暦閻㈠憡鍋╅柣鎴炆戦崣蹇撯攽閻樻彃鏆為柕鍥ㄧ箘閳ь剝顫夊ú锕傚磻婵犲倻鏆﹂柣鏃傗拡閺佸棝鏌嶈閸撶喖銆侀幘鏂ユ闁靛繆妾ч幏娲⒑閸涘﹦鈽夐柨鏇樺劤娴滄悂鎮介崨濠勫幈闂佺粯锚绾绢厽鏅堕悽鍛婄厽婵炴垵宕▍宥団偓瑙勬礃閿曘垽鍨鹃敃鍌氱闁绘劘灏欑槐锕傛⒒閸屾瑧绐旈柍褜鍓涢崑娑㈡嚐椤栨稒娅犻悗娑欙供濞堜粙鏌ｉ幇顓炵祷闁哄棙鐟╅弻宥囨嫚閺屻儱寮板Δ鐘靛仦閿曘垹鐣峰鈧幊鐘活敄閽樺澹曞銈嗘尵閸庢劕銆掓繝姘厪闁割偅绻冮ˉ婊冣攽椤斻劌鎳愮壕濂告煠閼圭増纭鹃柛鏃傛暬閺屾洟宕堕妸銉ヮ潙濡炪倧绠掗～澶愬箞閵婏妇绡€闁告洦鍘肩粭锟犳⒑閻熸澘妲婚柟铏悾鐑藉Ω閿斿墽鐦堥梺鍛婁緱閸ㄥジ寮ㄩ妸鈺傗拻闁稿本鐟чˇ锕傛煙绾板崬浜版鐐茬箳閳ь剨缍嗛崰鏍婵犳碍鐓曟繛鎴濆船閻忥紕鈧娲栧鍓佹崲濞戙垹绠ｆ繛鍡楃箳娴犲ジ姊虹紒妯虹瑨闁诲繑宀告俊鐢稿礋椤栨氨顔婇梺鐟扮摠缁诲秵绂掓ィ鍐┾拺缂備焦蓱鐏忔壆绱掔€ｎ偅宕岄柛鈹惧亾濡炪倖宸婚崑鎾绘煕濡崵鐭掔€规洘鍨块獮妯肩磼濡　鍋撻悽鍛婄叆婵犻潧妫濋妤€顭胯閸楁娊寮婚敐澶嬫櫇闁逞屽墴閹囧箻鐠囪尙鐣哄┑掳鍊曢崯顖炲窗閸℃稒鐓曢柡鍥ュ妼婢х増銇勯敂璇插籍婵﹥妞藉畷銊︾節鎼达絽濮搁梻浣规偠閸庢粓宕ㄩ鐐殿吅闂傚倸鍊烽懗鍫曘€佹繝鍥х妞ゅ繐妫涙稉宥夋煙鐎电啸闁活厼妫濋幃妤呮晲鎼粹剝鐏嶉梺缁樻尭缁绘劙鍩為幋锔藉€烽柤纰卞厸閾忓酣姊洪崨濠冣拹闁荤喆鍎甸垾锔炬崉閵婏箑鏋傞梺鍛婃处閸撴盯藝閵夆晜鈷戦悗鍦濞兼劙鏌涢妸銉﹀仴闁靛棔绀侀埢搴ㄥ箣閻愭彃骞戦梻鍌欑閻忔繈顢栭崱娑樻辈闁绘鐗呯换鍡涙煟閹板吀绨婚柍褜鍓氶悧鏇㈩敊韫囨梻绡€婵﹩鍓氭潏?闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧綊鏌熼梻瀵割槮缁炬儳缍婇弻鐔兼⒒鐎靛壊妲紒鐐劤缂嶅﹪寮婚敐澶婄闁挎繂鎲涢幘缁樼厱濠电姴鍊归崑銉╂煛鐏炶濮傜€殿噮鍣ｅ畷濂告偄閸涘鍞堕梻鍌欒兌椤牓顢栭崱娑樼闁告挆鍐ㄧ亰濡炪倖鎸鹃崑鎰ｉ崼鐔剁箚妞ゆ牗绻嶉崵娆愮箾閸涘洤娲﹂埛鎴炵箾閼奸鍤欐鐐搭殜閺岋綁鎮㈤崣澶嬬彋閻庢鍠栭…鐑藉箖閵忋倕绀傞柣鎾崇岸閸嬫挾绱掑Ο璇插伎濠德板€愰崑鎾翠繆椤愶絿鎳囨い銏☆焾閵囨劙骞掗幘顖涘闂佸搫顦遍崑鐐寸珶閸℃蛋鍥晜閹存帞绠氬銈嗗姧缁插潡骞婇崶褉鏀介柨娑樺閺嗩剛鈧娲滈崰鏍€佸☉姗嗘僵闁告鍋涚粻鐗堢節閻㈤潧浠╅柟娲讳簽瀵板﹪鎸婃径娑溾偓鍧楀级閸偄浜栧ù婊冪秺閺屾稑螖娴ｉ攱鍩汭ndex + nodeIndex

	nvvk::Buffer adjClusterInfoBuffer;

	nvvk::Buffer clusterPairInfoBuffer;
	nvvk::Buffer clusterPairGlobalInfoBuffer;
	nvvk::Buffer clusterPairHitInfoBuffer;

#if VRPG_Candidate_Samples_SampleMode == 1
	nvvk::Buffer candidatePoolInfoBuffer;
	nvvk::Buffer candidatePoolNodeInfoBuffer;
#endif

	nvvk::Buffer candidateSubTreeInfoBuffer;
	nvvk::Buffer candidateLeafNodeDataBuffer;

	nvvk::Buffer candidateSampleDataBuffer;
	nvvk::Buffer candidateSamplePrefixWSumBuffer;

	nvvk::Buffer candidateSampleIndexBuffer;
	nvvk::Buffer candidateSampleAABBBuffer;
	nvvk::Buffer candidateSampleData2Buffer;
	nvvk::Buffer candidateSampleDataHistoryBuffer;
	nvvk::Buffer spatialReservoirBuffer;
	nvvk::Buffer reservoirCurrentBuffer;
	nvvk::Buffer reservoirHistoryBuffer;

	nvvk::Buffer nearbyNodeTempInfoBuffer;
	nvvk::Buffer nearbyNodeInfoBuffer;

	bool spatialReuse = (VRPG_SPATIAL_REUSE != 0);
	bool temporalReuse = (VRPG_TEMPORAL_REUSE != 0);
	bool temporalReset = false;
	uint32_t temporalFrameIndex = 0;
	float temporalBlend = 0.1f;
	bool ditherSample = (VRPG_JITTER_SAMPLING != 0);
	bool rotateOutgoing = true;
private:
	OctreeCreateInfo_VRPG setting;

	nvvk::Buffer blockInfoBuffer_G;
	nvvk::Buffer blockInfoBuffer_E;
	nvvk::Buffer hasDataBlockIndexBuffer_G;
	nvvk::Buffer hasDataBlockIndexBuffer_E;
	nvvk::Buffer hasDataBlockCountBuffer;

	nvvk::Buffer divisibleNodeInfoBuffer_G;
	nvvk::Buffer threadGroupInfoBuffer;

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_initHasDataBlockInfo{};
	VkShaderEXT computeShader_getGlobalInfo{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};
	VkShaderEXT computeShader_initLightTreeLeaves{};
	VkShaderEXT computeShader_aggregateLightTree{};

	VkShaderEXT computeShader_getOctreeLabel1{};
	VkShaderEXT computeShader_getOctreeLabel2{};
	VkShaderEXT computeShader_getOctreeLabel3{};
	VkShaderEXT computeShader_getOctreeLabel4{};

	VkShaderEXT computeShader_initWeights{};
#if VRPG_Candidate_Samples_SampleMode == 1
	VkShaderEXT computeShader_getCandidateNodes_init{};
	VkShaderEXT computeShader_getCandidateNodes_1{};
	VkShaderEXT computeShader_getCandidateNodes_2{};
	VkShaderEXT computeShader_getCandidateNodes_3{};
	VkShaderEXT computeShader_getCandidateNodes_4{};
	VkShaderEXT computeShader_getCandidateNodes_5{};
#endif 
	VkShaderEXT computeShader_getCandidateNodes{};
	VkShaderEXT computeShader_dispatchGetCandidateLeafNodes{};
	VkShaderEXT computeShader_getCandidateLeafNodes{};
	VkShaderEXT computeShader_clusterPairHitTest{};
	VkShaderEXT computeShader_getCandidateNodeWeights{};;
	VkShaderEXT computeShader_buildSpatialReservoir{};
	VkShaderEXT computeShader_reuseReservoir{};
	VkShaderEXT computeShader_temporalReuse{};
	VkShaderEXT computeShader_createCandidatePrefixWSum{};;


	VkShaderEXT computeShader_getAdjClusterG{};

	VkShaderEXT computeShader_reservoirSample{};;
	VkShaderEXT computeShader_SpatialReuse{};;

	VkShaderEXT computeShader_dispatchGetNearbyClusterG{};
	VkShaderEXT computeShader_getNearbyClusterG1{};
	VkShaderEXT computeShader_getNearbyClusterG2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

	enum class GBuffers_Octree_VRPG {
		eGeometryClusterResult,
		eLightClusterResult,
		eGeometrySVO,
		eClusterPairResult,
		eBufferCount,
	};

	shaderio::OctreeDebugPushConstant_VRPG debugPushConstant;
	void geometryTreeDebug(VkCommandBuffer cmd);
	void lightTreeDebug(VkCommandBuffer cmd);
	void geometrySVODebug(VkCommandBuffer cmd);
	VkShaderEXT vertexShader_TreeDebug{};
	VkShaderEXT fragmentShader_TreeDebug{};

	bool changeClusterG = true;
	nvvk::Buffer clusterPairDebugGlobalInfoBuffer;
	shaderio::ClusterPairDebugPushConstant_VRPG debugPushConstant2;
	void clusterPairDebug(VkCommandBuffer cmd);
	VkShaderEXT computeShader_GetDebugClusterG{};
	VkShaderEXT vertexShader_ClusterPairDebug{};
	VkShaderEXT fragmentShader_ClusterPairDebug{};
	bool showNearbyClusters = false;
	int debugClusterIndex = -1;
};
}
#endif
