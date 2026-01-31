\ProvidesClass{tuhh_presentation}
% See also the wiki page https://collaborating.tuhh.de/e-4/tuhh_latex_presentation/-/wikis/tuhh_presentation-Class

% First option pass to base class (for options not declared by derived class)
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{beamer}}
\ProcessOptions\relax

\LoadClass[17pt]{beamer}
\usetheme{tuhh}
\setbeamertemplate{caption}[numbered]

\usepackage{.poppins_font/poppins}
\pdfmapfile{+.poppins_font/Poppins.map}

\usepackage[datesep=.,style=ddmmyy]{datetime2}

\AtBeginDocument{\usebeamerfont{normal text}}


%
% Getter and Setter Commands for Inserting Metadata
%

\def\university{\@dblarg\beamer@university}
\long\def\beamer@university[#1]#2{%
  \def\insertuniversity{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\street{\@dblarg\beamer@street}
\long\def\beamer@street[#1]#2{%
  \def\insertstreet{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\city{\@dblarg\beamer@city}
\long\def\beamer@city[#1]#2{%
  \def\insertcity{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\telephonenumber{\@dblarg\beamer@telephonenumber}
\long\def\beamer@telephonenumber[#1]#2{%
  \def\inserttelephonenumber{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\email{\@dblarg\beamer@email}
\long\def\beamer@email[#1]#2{%
  \def\insertemail{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\website{\@dblarg\beamer@website}
\long\def\beamer@website[#1]#2{%
  \def\insertwebsite{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

\def\closingstatement{\@dblarg\beamer@closingstatement}
\long\def\beamer@closingstatement[#1]#2{%
  \def\insertclosingstatement{\def\inst{\beamer@insttitle}\def\and{\beamer@andtitle}#2}%
}

% Redefine \includegraphics
\LetLtxMacro\latexincludegraphics\includegraphics
\renewcommand{\includegraphics}[2][]{%
  \textcolor{black}%
  {\latexincludegraphics[#1]{#2}}%
}

\renewcommand{\appendix}{%
  % 1. Hide future sections from TOC
  \addtocontents{toc}{\protect\setcounter{tocdepth}{0}}
  
  % 2. Create a clean "Appendix" Title Slide
  % We use 'plain' to avoid the dot logic, but manually apply theme colors
  \begin{frame}[plain, noframenumbering]
    \centering
    % Optional: If you want the background image, you can try adding it here:
    % \includegraphics[width=\paperwidth, height=\paperheight]{path/to/bg.png}
    
    \vspace*{0pt plus 1filll} % Vertical centering
    {
      \usebeamerfont{title}       % Use the standard title font
      \usebeamercolor[fg]{title}  % Use the standard title color (Teal)
      \Huge \textbf{Appendix}     % The Title
    }
    \vspace*{0pt plus 1filll}
  \end{frame}

  % 3. Reset counters for the appendix sections
  \setcounter{section}{0}
  \renewcommand{\thesection}{\Alph{section}}
}

%
% Defaults
%

\university{Hamburg University of Technology (TUHH)}
\closingstatement{Thank You very much}
\street{Am Schwarzenberg-Campus 1}
\city{21073 Hamburg}
\website{www.tuhh.de}
\telephonenumber{+49 40 42878-3330}
\institute{Corresponding Author's Institute}
\email{corresponding.author@tuhh.de}
\author[Corresponding Author]{First Author, Second Author}

\DeclareOption{english}{%
}
\DeclareOption{german}{%
  \closingstatement{Vielen Dank}%
  \university{Technische Universität Hamburg (TUHH)}%
  \settuhhwordmark{.theme_imgs/TUHH_logo-wortmarke_de_rgb.pdf}%
  \renewcommand{\figurename}{Abbildung}%
  \renewcommand{\tablename}{Tabelle}%
}

%
% Institute Branding 
%

% None
\DeclareOption{institute=none}{%
}

% Example
\DeclareOption{institute=test}{%
  \@ifclasswith{tuhh_presentation}{german}{
    \institute{Institutsname auf Deutsch}
  }{
    \institute{Institute Name in English}
  }
  \institutebranded{.theme_imgs/institute_logo_titlepage.pdf}{.theme_imgs/institute_logo_content.pdf}
}

% Institute of Communication Networks (E-4)
\DeclareOption{institute=e4}{%
  \@ifclasswith{tuhh_presentation}{german}{
    \institute{Institut für Kommunikationsnetze}
  }{
    \institute{Institute of Communication Networks}
  }
  \institutebranded{.theme_imgs/e4_logo_titlepage.pdf}{.theme_imgs/e4_logo_content.pdf}
}

% Institute of Mathematics (E-10)
\DeclareOption{institute=e10}{%
  \@ifclasswith{tuhh_presentation}{german}{
    \institute{Institut für Mathematik}
  }{
    \institute{Institute of Mathematics}
  }
  \institutebranded{.theme_imgs/e10_logo_titlepage.pdf}{.theme_imgs/e10_logo_content.pdf}
}

% Institute of Control Systems (E-14)
\DeclareOption{institute=e14}{%
  \@ifclasswith{tuhh_presentation}{german}{
    \institute{Institut für Regelungstechnik}
  }{
    \institute{Institute of Control Systems}
  }
  \institutebranded{.theme_imgs/e14_logo.pdf}{.theme_imgs/e14_logo.pdf}
}

% Second option pass to derived class
\ProcessOptions\relax
