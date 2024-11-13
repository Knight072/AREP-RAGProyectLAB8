import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://www.ig.com/latam/forex/que-es-forex-y-como-funciona",
               "https://www.checkpoint.com/es/cyber-hub/cyber-security/what-is-cybersecurity/#:~:text=What%20is%20Cyber%20Security%3F,activos%20contra%20las%20amenazas%20cibernéticas."),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=('gtie10', 'hydrated', 'swiftype', 'trackPerformanceTag', 'igcom', 'que-es-forex-y-como-funciona', 'dynamic-template-actual', 'theme--rebrand', 'rw-on', 'skip-to-content__link', 'esma-rw__wrapper', 'hide-esmaBtn__both', 'esma-mobile-font-size-small', 'esma-theme-light', 'esma-rw__content_wrap', 'full', 'esma-rw__content', 'full-text__warning', 'short-text__warning', 'esma-minimize__rw', 'main-parsys', 'grid', 'parsys', 'grid__bg', 'image-background', 'background-center-top', 'grid--desktop--12', 'grid--tablet--6', 'grid--mobile--1', 'full-height', 'grid__col', 'grid__col--desktop--12', 'grid__col--tablet--6', 'grid__col--mobile--1', 'desktop--order--0', 'tablet--order--0', 'mobile--order--0', 'desktop--align--auto', 'tablet--align--auto', 'mobile--align--auto', 'grid__col__inner', 'network-strip', 'parbase', 'nt-strip', 'nt-strip__inner', 'secondary-nav-only', 'nt-strip__content-left', 'nt-strip__dropdown-container', 'nt-strip__dropdown-toggle', 'nt-strip__current-site', 'icon', 'icon--dropdown-chevron', 'first-level', 'nt-strip__secondary-navigation', 'second-level', 'nt-strip__community', 'icon--community', 'nt-strip__academy', 'icon--academy', 'nt-strip__help-and-support', 'icon--help-and-support', 'nt-strip__content-right', 'nt-strip__account-inbox', 'icon--inbox', 'nt-strip__account-controls', 'nt-strip__prospect', 'logged-out-cta', 'cta', 'nt-strip__login-cta', 'nt-strip__signup-cta', 'nt-strip__account-dropdown', 'icon--account', 'nt-strip__account-name', 'nt-strip__account-information', 'myig_c', 'myig', 'account-myig', 'nt-strip__logout-container', 'nt-strip__account-logout', 'account-logout', 'header-nav', 'main-head', 'mega-menu', 'header__top', 'logo', 'default', 'header__navigation', 'mobile-menu-button', 'jsMenuMobile', 'hamburger-icon', 'screenreader', 'main-menu-container', 'jsMainMenu', 'mobile-actions', 'mobile-actions-back', 'visibility-hidden', 'mobile-actions-close', 'menu', 'first-level-menu', 'menu-item', 
                    'swiftype', 'sl_norewrite', 'optanon-category-C0003-C0004', 'optanon-category-C0003', 'optanon-category-C0002-C0003', 'optanon-category-C0002', 'cross', 'definition-template-default', 'single', 'single-definition', 'postid-185777', 'wp-schema-pro-2.7.10', 'tribe-js', 'lang-en-US', 'preload', 'smartling-es', 'no-print', 'cp_nav_menu__mobile__input', 'cp_top_nav_menu', 'header-link', 'cp_nav_menu', 'cp_search_geo', 'cp_menu__item', 'alt_menu_item', 'cp_menu__item__icon', 'cp_menu__item__search', 'cp_menu__item__main', 'cp_menu__item__ul', 'cp_menu__item__li', 'st-search-input', 'dark', 'cp_menu__item__geo', 'cp_menu__item__li__inner_li', 'lang-item-en', 'lang-item-es', 'lang-item-fr', 'lang-item-de', 'lang-item-it', 'lang-item-pt', 'lang-item-jp', 'lang-item-cn', 'lang-item-kr', 'lang-item-tw', 'cp_menu', 'mobile_cp_menu', 'cp_menu__desktop', 'cp_menu__item__li_bg', 'cp_menu__tabs', 'cp_menu__tabbed', 'cp-nav-solutions-1', 'active', 'cp-nav-solutions-2', 'cp-nav-solutions-3', 'cp-nav-solutions-sub-1', 'cp-nav-inner-sub', 'cp_menu__item__li_mw', 'cp_menu__item__ul_split', 'cp_menu__item__li_cta', 'btn', 'btn-primary', 'cp-nav-solutions-sub-2', 'cp-nav-solutions-sub-3', 'cp-nav-platform-1', 'cp-nav-platform-2', 'cp-nav-platform-3', 'cp-nav-platform-4', 'cp-nav-platform-5', 'cp-nav-platform-sub-1', 'cp_menu__item__li__title', 'cp_menu__item__border', 'infinity-desc', 'cp-nav-platform-sub-2', 'quantum-desc', 'cp-nav-platform-sub-3', 'cloudguard-desc', 'cp-nav-platform-sub-4', 'harmony-desc', 'cp-nav-platform-sub-5', 'core-desc', 'cp-nav-support-1', 'cp-nav-support-2', 'cp-nav-support-3', 'cp-nav-support-4', 'cp-nav-support-5', 'cp-nav-support-6', 'cp-nav-support-sub-1', 'assess-desc', 'cp_menu__no_link', 'cp_menu__item__li__desktop_link', 'cp_menu__more', 'cp-nav-support-sub-2', 'transform-desc', 'cp-nav-support-sub-3', 'master-desc', 'cp-nav-support-sub-4', 'respond-desc', 'cp-nav-support-sub-5', 'manage-desc', 'cp-nav-support-sub-6', 'support-desc', 'cp_menu__mobile', 'cp_menu__item__li_cta_align', 'cp_nav_menu__logo', 'cp_search_geo__mobile', 'cp_nav-icon', 'sr-only', 'cp_nav-icon__inside', 'smartling_body', 'cyberhub-bread', 'container', 'breadcrumb', 'breadcrumb-item', 'row', 'vertical-center', 'col-md-8', 'btn-lg', 'btn-secondary', 'col-md-4', 'align-center', 'navbar', 'navbar-default', 'col-md-12', 'css-nav', 'menu-btn', 'menu-icon', 'navicon', 'menu', 'col-md-push-4', 'cyberhub-section', 'anchor', 'white', 'grey', 'col-md-pull-8', 'trd-ph-embedded', 'trd-unit-populated', 'trd_comp_330f1678-626f-44a8-be02-39a5dc8e1632', 'trd-recommend-loading', 'trd-comp-container', 'padded', 'trd-close-btn', 'embedded', 'component-content', 'close-cont', 'external-close-cont', 'external', 'close-btn', 'container-elements', 'internal-close-cont', 'internal', 'int-con-td-item', 'title-container', 'trd-title-wing', 'trd-title', 'sub-title', 'linkable-container', 'sidebar-slide', 'sidebar-link', 'close', 'icon', 'message', 'chat-hello', 'iframe-contain', 'hidden-print', 'close-x', 'feedback-icon', 'modal', 'fade', 'modal-dialog', 'modal-content', 'modal-body', 'imagepreview', 'site-footer', 'refresh-2024', 'container-fluid', 'blocks', 'cp_footer_menu', 'cp_footer_menu__item', 'y-inline', 'cp_footer_menu__item__title', 'cp_footer_menu__item__inner_menu', 'cp_footer_menu__item__inner_menu_link', 'footer-link', 'n-inline', 'contact', 'contact__title', 'contact__loc', 'min-footer', 'col-lg-5', 'col-lg-push-7', 'text-right', 'followUs', 'title', 'fa', 'fa-facebook-square', 'fa-linkedin', 'fa-youtube-play', 'col-lg-7', 'col-lg-pull-5', 'text-left', 'tagline', 'tagline_tm', 'copyright', 'col-md-11', 'col-md-1', 'YouTubeModal', 'modal-header', 'modal-title', 'embed-responsive', 'embed-responsive-4by3', 'onetrust-pc-dark-filter', 'ot-hide', 'ot-fade-in', 'otFloatingRounded', 'otRelFont', 'ot-bottom-left', 'ot-wo-title', 'vertical-align-content', 'ot-sdk-container', 'ot-sdk-row', 'ot-sdk-twelve', 'ot-sdk-columns', 'onetrust-banner-options', 'cookie-setting-link', 'banner_logo', 'onetrust-close-btn-handler', 'onetrust-close-btn-ui', 'banner-close-button', 'ot-close-icon', 'otPcCenter', 'ot-pc-header', 'ot-pc-logo', 'ot-pc-scrollbar', 'ot-optout-signal', 'ot-optout-icon', 'privacy-notice-link', 'ot-cat-grp', 'ot-accordion-layout', 'ot-cat-item', 'ot-vs-config', 'ot-acc-hdr', 'ot-plus-minus', 'ot-cat-header', 'ot-tgl', 'category-switch-handler', 'ot-switch', 'ot-switch-nob', 'ot-label-txt', 'ot-acc-grpcntr', 'ot-acc-txt', 'ot-acc-grpdesc', 'ot-category-desc', 'ot-always-active-group', 'ot-always-active', 'ot-hosts-ui', 'ot-link-btn', 'back-btn-handler', 'ot-lst-subhdr', 'ot-search-cntr', 'ot-scrn-rdr', 'ot-fltr-cntr', 'ot-fltr-scrlcnt', 'ot-fltr-opts', 'ot-fltr-opt', 'ot-chkbox', 'category-filter-handler', 'ot-label-status', 'ot-fltr-btns', 'ot-host-cnt', 'ot-sel-all', 'ot-sel-all-hdr', 'ot-consent-hdr', 'ot-li-hdr', 'ot-sel-all-chkbox', 'ot-sdk-column', 'ot-pc-footer', 'ot-btn-container', 'ot-pc-refuse-all-handler', 'save-preference-btn-handler', 'ot-pc-footer-logo', 'ot-text-resize', 'trd_comp_8af35c59-a1cf-4cbc-8cdb-f5c9ce766018', 'single-img', 'light', 'position_4', 'trd-link', 'trd-img', 'trd-img-link', 'interactable-container', 'trd-description', 'linkable-items', 'drift-conductor-item', 'drift-frame-chat', 'drift-frame-chat-align-right', 'drift-has-chat', 'drift-frame-controller', 'drift-frame-controller-align-right')
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_rag_response(question: str) -> str:
    return rag_chain.invoke(question)
