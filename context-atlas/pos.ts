/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/** Part of speech tags and descriptions, from http://www.nltk.org/*/
export interface POSTag {
  tag: string;
  description: string;
  dispPos?: string;
}

export const POS: POSTag[] =
    [
      {"tag": "APPRART", "description": "APPRART", "dispPos": "AP"},
      {"tag": "APZR", "description": "APZR", "dispPos": "AP"},
      {"tag": "ART", "description": "ART", "dispPos": "AR"},
      {"tag": "CARD", "description": "CARD", "dispPos": "CA"},
      {"tag": "FM", "description": "FM", "dispPos": "FM"},
      {"tag": "KOKOM", "description": "KOKOM", "dispPos": "KO"},
      {"tag": "KON", "description": "KON", "dispPos": "KO"},
      {"tag": "KOUI", "description": "KOUI", "dispPos": "KO"},
      {"tag": "KOUS", "description": "KOUS", "dispPos": "KO"},
      {"tag": "NE", "description": "NE", "dispPos": "NE"},
      {"tag": "NN", "description": "NN", "dispPos": "NN"},
      {"tag": "PAV", "description": "PAV", "dispPos": "PA"},
      {"tag": "PDAT", "description": "PDAT", "dispPos": "PD"},
      {"tag": "PDS", "description": "PDS", "dispPos": "PD"},
      {"tag": "PIAT", "description": "PIAT", "dispPos": "PI"},
      {"tag": "PIS", "description": "PIS", "dispPos": "PI"},
      {"tag": "PPER", "description": "PPER", "dispPos": "PP"},
      {"tag": "PPOSAT", "description": "PPOSAT", "dispPos": "PP"},
      {"tag": "PRELAT", "description": "PRELAT", "dispPos": "PR"},
      {"tag": "PRELS", "description": "PRELS", "dispPos": "PR"},
      {"tag": "PRF", "description": "PRF", "dispPos": "PR"},
      {"tag": "PTKA", "description": "PTKA", "dispPos": "PT"},
      {"tag": "PTKANT", "description": "PTKANT", "dispPos": "PT"},
      {"tag": "PTKNEG", "description": "PTKNEG", "dispPos": "PT"},
      {"tag": "PTKVZ", "description": "PTKVZ", "dispPos": "PT"},
      {"tag": "PTKZU", "description": "PTKZU", "dispPos": "PT"},
      {"tag": "PWAT", "description": "PWAT", "dispPos": "PW"},
      {"tag": "PWAV", "description": "PWAV", "dispPos": "PW"},
      {"tag": "PWS", "description": "PWS", "dispPos": "PW"},
      {"tag": "TRUNC", "description": "TRUNC", "dispPos": "TR"},
      {"tag": "VAFIN", "description": "VAFIN", "dispPos": "VA"},
      {"tag": "VAINF", "description": "VAINF", "dispPos": "VA"},
      {"tag": "VAPP", "description": "VAPP", "dispPos": "VA"},
      {"tag": "VMFIN", "description": "VMFIN", "dispPos": "VM"},
      {"tag": "VMINF", "description": "VMINF", "dispPos": "VM"},
      {"tag": "VVFIN", "description": "VVFIN", "dispPos": "VV"},
      {"tag": "VVIMP", "description": "VVIMP", "dispPos": "VV"},
      {"tag": "VVINF", "description": "VVINF", "dispPos": "VV"},
      {"tag": "VVIZU", "description": "VVIZU", "dispPos": "VV"},
      {"tag": "VVPP", "description": "VVPP", "dispPos": "VV"},
    ]


    export const SimplePOS: POSTag[] = [
      {"tag": "AD", "description": "AD"},
      {"tag": "AP", "description": "AP"},
      {"tag": "AR", "description": "AR"},
      {"tag": "CA", "description": "CA"},
      {"tag": "FM", "description": "FM"},
      {"tag": "KO", "description": "KO"},
      {"tag": "NE", "description": "NE"},
      {"tag": "NN", "description": "NN"},
      {"tag": "PA", "description": "PA"},
      {"tag": "PD", "description": "PD"},
      {"tag": "PI", "description": "PI"},
      {"tag": "PP", "description": "PP"},
      {"tag": "PR", "description": "PR"},
      {"tag": "PT", "description": "PT"},
      {"tag": "PW", "description": "PW"},
      {"tag": "TR", "description": "TR"},
      {"tag": "VA", "description": "VA"},
      {"tag": "VM", "description": "VM"},
      {"tag": "VV", "description": "VV"}
    ]
